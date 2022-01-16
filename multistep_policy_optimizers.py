import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.nn import Conv1d
import copy
import collections

from utils import one_hot_embedding, temperature_search
from policy_optimizers import Optimizer
from episode_buffers import EpisodeExperienceBuffer

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class MultiStep_Second_Level_SAC_PolicyOptimizer(Optimizer):
    def __init__(
        self, learn_alpha=True, batch_size=32, 
        discount_factor=0.999, clip_value=1.0, 
        n_actions=4, init_epsilon=1.0, min_epsilon=0.4, 
        delta_epsilon=2.5e-7, entropy_factor=0.95,
        weight_q_loss = 0.05, entropy_update_rate=0.05, 
        alpha_v_weight=0.1, restrain_policy_update=False,
        clip_q_error=False, clip_value_q_error=None,
        use_H_mean=True, use_entropy=True,
        normalize_q_error=False, normalize_q_dist=False,
        target_update_rate=5e-3, 
        state_dependent_temperature=False,
        actor_loss_function='kl',
        c_minus_temp_search = 1e-2,
        log_novelty_min = -10,
        log_novelty_max = -4,
        discount_factor_int=0.99,
        int_heads=False
        ):
        super().__init__()

        self.learn_alpha = learn_alpha
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.discount_factor_int = discount_factor_int
        self.clip_value = clip_value
        self.entropy_target = np.log(n_actions)
        self.epsilon = init_epsilon
        self.delta_epsilon = delta_epsilon
        self.min_epsilon = entropy_factor
        self.weight_q_loss = weight_q_loss
        self.alpha_v_weight = alpha_v_weight
        self.entropy_update_rate = entropy_update_rate
        self.target_update_rate = target_update_rate
        self.restrain_policy_update = restrain_policy_update
        self.clip_q_error = clip_q_error
        self.clip_value_q_error = clip_value_q_error
        self.normalize_q_error = normalize_q_error
        self.use_entropy = use_entropy
        self.normalize_q_dist = normalize_q_dist
        self.use_H_mean = use_H_mean
        self.H_mean = None
        self.state_dependent_temperature = state_dependent_temperature
        self.actor_loss_function = actor_loss_function
        self.c_minus_temp_search = c_minus_temp_search
        self.log_novelty_min = log_novelty_min
        self.log_novelty_max = log_novelty_max
        self.int_heads = int_heads
    
    def optimize(
        self, 
        agents: list, 
        database: EpisodeExperienceBuffer, 
        n_step_td: int = 1
        ) -> None: 

        if database.__len__() < 1:
            return None
        
        # Sample batch
        states, actions, ext_rewards, \
            dones, dists = \
            database.sample(self.batch_size, n_step=n_step_td)      

        q_target = 0.0
        log_softmax_target = 0.0
        HA_s_mean = 0.0

        # TODO: handle more than 1 agent
        agent = copy.deepcopy(agents[-1])
        n_actions = agent._n_actions

        # Calculate RND loss and novelty
        log_novelty, int_rewards = agent.calc_novelty(
            states[:,1:,:,:,:].reshape(-1, *states.shape[-3:]))
        int_rewards = int_rewards.reshape(ext_rewards.shape)
        
        # Alias for actor-critic module
        actor_critic = agent.second_level_architecture

        # Calculate q-values and action likelihoods
        q, q_target_0, PA_s, log_PA_s, log_alpha_nostate = \
            actor_critic(states.reshape(-1, *states.shape[-3:]))

        if not actor_critic._parallel:
            q[0] = q[0].reshape(*states.shape[:-2],-1)
            q[1] = q[1].reshape(*states.shape[:-2],-1)
            q_target_0[0] = q_target_0[0].reshape(
                *states.shape[:2],-1, n_actions)
            q_target_0[1] = q_target_0[1].reshape(
                *states.shape[:2],-1, n_actions)
        else:
            q = q.reshape(
                *states.shape[:2],*q.shape[-3:])
            q_target_0 = q_target_0.reshape(
                *states.shape[:2],*q_target_0.shape[-3:])

        PA_s = PA_s.reshape(*states.shape[:2],-1)
        log_PA_s = log_PA_s.reshape(*states.shape[:2],-1)

        # Calculate entropy of the action distributions
        HA_s = -(PA_s * log_PA_s).sum(-1, keepdim=True)
        HA_s_mean = HA_s.detach().mean()

        # Update mean entropy
        if self.H_mean is None:
            self.H_mean = HA_s_mean.item()
        else:
            self.H_mean = (
                HA_s_mean.item() * self.entropy_update_rate 
                + self.H_mean * (1.0-self.entropy_update_rate)
            )
        
        # Choose minimum q-value to avoid overestimation of target
        if not actor_critic._parallel:
            q_min = torch.min(q_target_0[0], q_target_0[1])
        else:
            q_min = q_target_0.min(-2)[0]

        # Set temperature
        if self.state_dependent_temperature:
            desired_entropy = torch.exp(
                (log_novelty.clamp(
                    self.log_novelty_min, 
                    self.log_novelty_max
                    ) 
                - self.log_novelty_max
                ) / (np.abs(self.log_novelty_max - self.log_novelty_min)+1e-6)
            ) * self.entropy_target * 0.995
            with torch.no_grad():
                log_alpha_state, n_iter = temperature_search(
                    q_min[:,:,0,:].detach().reshape(-1, n_actions), desired_entropy, 
                    c_minus=self.c_minus_temp_search
                    )
                alpha = torch.exp(log_alpha_state).reshape(*q_min.shape[:2],1)
        else:
            alpha = log_alpha_nostate.exp().item()
            n_iter = 0

        # Calculate next v-value, exactly, with the next action distribution
        next_v_target_ext = (PA_s.detach() * q_min[:,:,0,:])[:,1:,:].sum(-1)

        # next_v_target_ext = (PA_s.detach() * (
        #     q_min[:,:,0,:] - alpha * float(self.use_entropy) * (
        #         log_PA_s.detach() + float(self.use_H_mean) * self.H_mean
        #         )
        #     ))[:,1:,:].sum(-1)
        
        next_v_target_int = (
            PA_s.detach() * q_min[:,:,1,:]
            )[:,1:,:].sum(-1)

        # Estimate q-value target by sampling Bellman expectation
        # TODO: Importance Sampling
        ext_rewards += self.discount_factor * (PA_s.detach() * (
            - alpha * float(self.use_entropy) * (
                log_PA_s.detach() + float(self.use_H_mean) * self.H_mean
                )
            ))[:,1:,:].sum(-1).detach()
        
        next_v_target_valid_ext = (next_v_target_ext * (1.-dones))

        if n_step_td > 1: 
            padded_rewards_ext = torch.cat(
                [ext_rewards, torch.zeros_like(ext_rewards)[:,:-1]], 
                dim=1
            )
            padded_rewards_int = torch.cat(
                [int_rewards, torch.zeros_like(int_rewards)[:,:-1]], 
                dim=1
            )
            padded_next_v_target_ext = torch.cat(
                [
                    next_v_target_valid_ext, 
                    torch.zeros_like(next_v_target_valid_ext)[:,:-1]
                ], 
                dim=1
            )
            padded_next_v_target_int = torch.cat(
                [
                    next_v_target_int, 
                    torch.zeros_like(next_v_target_int)[:,:-1]
                ], 
                dim=1
            )
        else:
            padded_rewards_ext = ext_rewards
            padded_rewards_int = int_rewards
            padded_next_v_target_ext = next_v_target_valid_ext
            padded_next_v_target_int = next_v_target_int

        discounted_sum_of_rewards_ext = [ext_rewards]
        for i in range(1, n_step_td):
            partial_sum_rewards_ext = discounted_sum_of_rewards_ext[-1]
            rewards_next_ext = padded_rewards_ext[:,i:i+n_step_td]
            rewards_sum_ext = (
                partial_sum_rewards_ext 
                + self.discount_factor**i * rewards_next_ext
            )
            discounted_sum_of_rewards_ext.append(rewards_sum_ext)
        
        discounted_sum_of_rewards_int = [int_rewards]
        for i in range(1, n_step_td):
            partial_sum_rewards_int = discounted_sum_of_rewards_int[-1]
            rewards_next_int = padded_rewards_int[:,i:i+n_step_td]
            rewards_sum_int = (
                partial_sum_rewards_int 
                + self.discount_factor_int**i * rewards_next_int
            )
            discounted_sum_of_rewards_int.append(rewards_sum_int)

        q_target_mixed_steps_ext_list = []
        for i in range(1, n_step_td+1):
            sum_ext = discounted_sum_of_rewards_ext[i-1]
            value_ = padded_next_v_target_ext[:,i-1:i-1+n_step_td]
            q_target_i_steps = (
                sum_ext 
                + self.discount_factor**i * value_
            )
            q_target_mixed_steps_ext_list.append(q_target_i_steps)

        q_target_mixed_steps_int_list = []
        for i in range(1, n_step_td+1):
            sum_int = discounted_sum_of_rewards_int[i-1]
            value_ = padded_next_v_target_int[:,i-1:i-1+n_step_td]
            q_target_i_steps = (
                sum_int 
                + self.discount_factor_int**i * value_
            )
            q_target_mixed_steps_int_list.append(q_target_i_steps)

        q_target_mixed_steps_ext = torch.stack(
            q_target_mixed_steps_ext_list, dim=2
        )

        q_target_mixed_steps_int = torch.stack(
            q_target_mixed_steps_int_list, dim=2
        )

        q_target_mixed_steps = torch.stack(
            [q_target_mixed_steps_ext, q_target_mixed_steps_int],
            dim=2
        )

        # Importance sampling ratios
        off_action_lklhood = PA_s[:,:-1,:].detach().reshape(-1,n_actions)[
            np.arange(self.batch_size*n_step_td), np.concatenate(actions)
        ]
        on_action_lklhood = dists.reshape(-1,n_actions)[
            np.arange(self.batch_size*n_step_td), np.concatenate(actions)
        ]
        IS_step_ratios = (
            off_action_lklhood / on_action_lklhood
        ).reshape(-1,n_step_td)
        IS_trajectory_ratios = []
        for h in range(0, n_step_td):
            IS_trajectory_ratios_h = [IS_step_ratios[:,h]]
            for i in range(1, n_step_td-h-1):
                IS_trajectory_ratios_h.append(
                    IS_trajectory_ratios_h[i-1] * IS_step_ratios[:,h+i]
                )
            ones_ = torch.ones_like(IS_step_ratios[:,h])
            zeros_ = torch.zeros_like(IS_step_ratios[:,h])
            IS_trajectory_ratios_h = (
                [ones_] 
                + int((h+1) != n_step_td) * IS_trajectory_ratios_h 
                + h * [zeros_]
            )
            IS_trajectory_ratios_h = torch.stack(
                IS_trajectory_ratios_h, dim=1)
            IS_trajectory_ratios.append(IS_trajectory_ratios_h)
        IS_trajectory_ratios = torch.stack(
                IS_trajectory_ratios, dim=1)
        IS_test = IS_trajectory_ratios[0,:,:]
        IS_trajectory_ratios /= IS_trajectory_ratios.sum()     

        # triu_matrix = torch.flip(
        #     torch.triu(
        #         torch.ones(n_step_td, n_step_td).to(device)
        #     ), 
        #     dims=[1]
        # )
        # q_target_unnormalized = torch.einsum(
        #     'ijk,jk->ij', 
        #     q_target_mixed_steps,
        #     triu_matrix
        # )
        # scale_ = torch.linspace(
        #     n_step_td,1,n_step_td
        # ).reshape(1,-1).to(device)
        # q_target = (q_target_unnormalized / scale_).unsqueeze(2)
        
        if not actor_critic._parallel:
            # Select q-values corresponding to the action taken
            q1_A = q[0][:,:-1,:,:].reshape(-1,n_actions)[
                np.arange(self.batch_size*n_step_td), 
                np.concatenate(actions)
                ].reshape(-1,n_step_td)
            q2_A = q[1][:,:-1,:,:].reshape(-1,n_actions)[
                np.arange(self.batch_size*n_step_td), 
                np.concatenate(actions)
                ].reshape(-1,n_step_td)

            # Calculate losses for both critics as the quadratic TD errors
            q1_loss = (q1_A - q_target.detach()).pow(2).mean()
            q2_loss = (q2_A - q_target.detach()).pow(2).mean()
            q_loss = q1_loss + q2_loss
        else:
            # Select q-values corresponding to the action taken
            q_A = q[:,:-1,:,:,:].reshape(-1,*q.shape[-3:])[
                np.arange(self.batch_size*n_step_td),:,:,np.concatenate(actions)
                ].reshape(q.shape[0],-1,*q.shape[-3:-1])

            # Calculate losses for both critics as the quadratic TD errors
            max_q_dif = max(
                (
                    q_A.unsqueeze(4) 
                    - q_target_mixed_steps.unsqueeze(3)
                ).abs().max().item(), 
                1e-6
            )
            if self.normalize_q_error:
                q_div = max_q_dif
            else:
                q_div = 1.0

            if self.clip_q_error: # TODO
                q_loss = ((q_A - q_target.detach())/max_q_dif).clip(
                    -self.clip_value_q_error,self.clip_value_q_error
                    ).pow(2).mean()
            else:
                q_loss = (
                    (
                        (
                            q_A.unsqueeze(4) 
                            - q_target_mixed_steps.unsqueeze(3).detach()
                        ) / q_div
                    ).pow(2) * IS_trajectory_ratios.unsqueeze(2).unsqueeze(3).detach()
                ).sum()

        # Optimize critic
        actor_critic.q.optimizer.zero_grad()
        q_loss.backward()
        clip_grad_norm_(actor_critic.q.parameters(), self.clip_value)
        actor_critic.q.optimizer.step()    
        
        # Calculate q-values and action likelihoods after critic SGD
        q, PA_s, log_PA_s = actor_critic.evaluate_actor(
            states[:,:-1,:,:,:].reshape(-1, *states.shape[-3:])
        )
        if q.shape[1] == 2:
            q = 2.0*q[:,0,:,:] + q[:,1,:,:]
        else:
            q = q.squeeze(1)

        if not actor_critic._parallel:
            q[0] = q[0].reshape(states.shape[0],-1,n_actions)
            q[1] = q[1].reshape(states.shape[0],-1,n_actions)
        else:
            q = q.reshape(states.shape[0],-1,*q.shape[-2:])

        # Choose mean q-value to avoid overestimation
        if not actor_critic._parallel:            
            q_dist = torch.min(q[0], q[1]).reshape(-1,n_actions)
        else:
            q_dist = q.min(-2)[0].reshape(-1,n_actions)

        # TODO: check validity
        if self.restrain_policy_update:
            q_errors = (
                q_A - q_target.unsqueeze(1).detach()
                ).pow(2).sum(1, keepdim=True).detach()
            q_dist = (
                q_dist + alpha * q_errors * log_PA_s) / (1.0 + q_errors)

        q_dist = (
                q_dist - q_dist.max(-1, keepdim=True)[0]
            )

        if self.normalize_q_dist:
            q_dist = (
                q_dist - q_dist.mean(-1, keepdim=True)
                ) / (q_dist.std(-1, keepdim=True) + 1e-6)

        # Calculate state-dependent temperature for current state
        if self.state_dependent_temperature:
            log_novelty_state = agent.calc_novelty(
                states[:,:-1,:,:,:].reshape(-1, *states.shape[-3:]))[0]
            desired_entropy = torch.exp(
                (log_novelty_state.clamp(
                    self.log_novelty_min, 
                    self.log_novelty_max
                    ) 
                - self.log_novelty_max
                ) / (np.abs(self.log_novelty_max - self.log_novelty_min)+1e-6)
            ) * self.entropy_target * 0.995
            with torch.no_grad():
                log_alpha_state, n_iter = temperature_search(
                    q_dist.detach(), desired_entropy, 
                    c_minus=self.c_minus_temp_search
                    )
                alpha = torch.exp(log_alpha_state).reshape(-1,1) 

        # Calculate normalizing factors for target softmax distributions
        z = torch.logsumexp(q_dist/(alpha+1e-10), -1, keepdim=True)

        # Calculate the target log-softmax distribution
        log_softmax_target = q_dist/(alpha+1e-10) - z
        softmax_target = torch.exp(q_dist)**(1/(alpha+1e-10))
        softmax_target /= softmax_target.sum(-1, keepdim=True)
        entropy_target = -(
            softmax_target * log_softmax_target
            ).sum(-1).mean()
        
        # Calculate actor losses as the KL divergence between action 
        # distributions and softmax target distributions
        difference_ratio = (log_PA_s - log_softmax_target.detach())
        kl_div_PA_s_target = (PA_s * difference_ratio).sum(
            -1, keepdim=True).mean()

        if self.actor_loss_function == 'kl':
            actor_loss = kl_div_PA_s_target
        elif self.actor_loss_function == 'jeffreys':
            kl_div_target_PA_s = -(
                softmax_target.detach() * difference_ratio
                ).sum(-1, keepdim=True).mean()
            actor_loss = kl_div_PA_s_target + kl_div_target_PA_s
        else:
            raise ValueError(
                'Invalid actor loss function. ' 
                + 'Should be kl or jeffreys divergence.')

        # Optimize actor
        actor_critic.actor.optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(actor_critic.actor.parameters(), self.clip_value)
        actor_critic.actor.optimizer.step()

        if self.state_dependent_temperature:
            alpha_error = (HA_s_mean - desired_entropy)**2
            alpha_loss = ((alpha_error.mean())**0.5).detach()
            mean_alpha = alpha.mean().item()
        else:
            # Calculate loss for temperature parameter alpha 
            scaled_min_entropy = self.entropy_target * self.epsilon
            alpha_error = (HA_s_mean - scaled_min_entropy).mean()
            alpha_loss = log_alpha_nostate * alpha_error.detach()

            # Optimize temperature (if it is learnable)
            if self.learn_alpha:
                # Create optimizer and optimize model
                actor_critic.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                clip_grad_norm_([actor_critic.log_alpha], self.clip_value)
                actor_critic.alpha_optimizer.step()       

            mean_alpha = alpha 

        # Update targets of actor-critic and temperature param.
        actor_critic.update(self.target_update_rate)

        # Anneal epsilon
        self.epsilon = np.max([self.epsilon - self.delta_epsilon, self.min_epsilon])   

        agents.append(agent) 

        metrics = {'q_loss': q_loss.item(),
                    'actor_loss': actor_loss.item(),
                    'alpha_loss': alpha_loss.item(),
                    'SAC_epsilon': self.epsilon,
                    'alpha': mean_alpha,
                    'base_entropy': self.H_mean,
                    'target_entropy': entropy_target.item(),
                    'n_iter_temp_search': n_iter,
                    'log_novelty_mean': log_novelty.mean().item(),
                    'log_novelty_std': log_novelty.std().item(),
                    'int_rewards_mean': int_rewards.mean().item(),
                    'int_rewards_std': int_rewards.std().item(),
                    'max_q_diff': max_q_dif
                    }

        return metrics
