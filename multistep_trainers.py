from collections import  deque
import numpy as np
from episode_buffers import (
    EpisodeExperienceBuffer, ExperienceFirstLevel)
from multistep_policy_optimizers import MultiStep_Second_Level_SAC_PolicyOptimizer as PolicyOptimizer

from utils import cat_state_task, scale_action

import wandb

import cv2
video_folder = '/home/researcher/Diego/CLminihack/videos/'


class MutiStep_Second_Level_Trainer:
    def __init__(self, optimizer_kwargs={}):
        self.optimizer = PolicyOptimizer(**optimizer_kwargs)
        
    def loop(self, env, agents, database, n_episodes=10, train=True,
            max_episode_steps=2000, train_each=1, update_database=True, 
            render=False, store_video=False, wandb_project=False, 
            save_model=True, save_model_each=50, MODEL_PATH='', 
            save_step_each=2, greedy_sampling=False, n_step_td=2, eval_each=5, 
            init_sac=True, init_rnd=True, init_steps_sac=500):

        best_return = -np.infty

        # Start video recording
        if store_video:
            video_filename = (
                video_folder
                + env.spec.id+str(agents[-1].get_id())
                + '.mp4'
            )
            fourcc = cv2.VideoWriter_fourcc(*'avc1')#*'avc1'
            video = cv2.VideoWriter(video_filename, fourcc, 4, (1264, 336))

        # Set alias for RND module
        rnd_module = agents[-1].rnd_module
        
        # Initialize database and statistics
        init = init_sac or init_rnd
        if init:
            self.initialize(
                env, agents, database, n_step_td,
                max_episode_steps, init_steps_sac, init_sac, init_rnd)

        returns = []
        for episode in range(0, n_episodes):
            state_buffer = deque(maxlen=max_episode_steps)
            action_buffer = deque(maxlen=max_episode_steps)
            reward_buffer = deque(maxlen=max_episode_steps)
            done_buffer = deque(maxlen=max_episode_steps)
            dist_buffer = deque(maxlen=max_episode_steps) 
            
            step_counter = 0
            episode_done = False
            state = env.reset()
            episode_return = 0.0

            state_buffer.append(state['pixel'])

            while not episode_done:
                action, dist = agents[-1].sample_action(
                    state, explore=(train or (not greedy_sampling)))
                
                if render:
                    env.render()
                
                next_state, reward, done, info = env.step(action)
                dist = (dist + 1e-6) / (dist + 1e-6).sum()

                action_buffer.append(action)
                reward_buffer.append(reward)
                done_buffer.append(done)
                dist_buffer.append(dist)              
                
                if store_video:
                    img = state['original']
                    video.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                episode_return += reward
                state = next_state.copy()
                state_buffer.append(state['pixel'])
                                
                step_counter += 1

                # Train agent
                should_train_in_this_step = train and ((step_counter % train_each) == 0) 
                if should_train_in_this_step:
                    metrics = self.optimizer.optimize(agents, database, n_step_td)
                    #agents.append(agent_last)                    
                    if wandb_project and metrics is not None:
                        metrics['step'] = step_counter
                        wandb.log(metrics)

                # Finish episode
                if (step_counter >= max_episode_steps) or done:
                    episode_done = True
            returns.append(episode_return)
                
            # Train RND module
            train_rnd = train and (rnd_module is not None)
            if train_rnd:
                # Update RND observation statistics
                state_batch = np.stack(state_buffer).astype(np.float)/255.
                rnd_module.obs_rms.update(state_batch)

                # Update predictor parameters
                rnd_loss, int_rewards = agents[-1].train_rnd_module(
                    state_batch)
                
                 # Update RND reward statistics
                int_pseudo_returns = [
                    rnd_module.discounted_reward.update(reward.item())
                    for reward in int_rewards]
                int_pseudo_returns = np.stack(int_pseudo_returns)
                rnd_module.reward_rms.update_from_moments(
                    np.mean(int_pseudo_returns),
                    np.std(int_pseudo_returns)**2,
                    len(int_pseudo_returns)
                )
                int_reward_mean = rnd_module.reward_rms.mean
                int_reward_std = rnd_module.reward_rms.var**0.5
            else:
                rnd_loss = False
                int_reward_mean = int_reward_std = 0.0

            if update_database:
                done_buffer[-1] = True
                episode_trajectory = ExperienceFirstLevel(
                    state_buffer, action_buffer, reward_buffer, 
                    done_buffer, dist_buffer
                )
                database.append(episode_trajectory)                    

            # Log return
            if wandb_project and train:
                wandb.log(
                    {
                        'episode': episode, 
                        'return': episode_return, 
                        'rnd_loss': rnd_loss,
                        'int_reward_mean': int_reward_mean,
                        'int_reward_std': int_reward_std,
                    }
                )

            # Save model
            if save_model and ((episode + 1) % save_model_each == 0):
                agents[-1].save(MODEL_PATH + env.spec.id + '/')
                        
            if train and (episode_return > best_return):
                best_return = episode_return
                agents[-1].save(MODEL_PATH + env.spec.id + '/', best=True)
            
            # Eval agent
            if train and ((episode+1) % eval_each == 0):
                eval_returns = self.loop(env, agents, None, n_episodes=1, train=False, 
                    max_episode_steps=max_episode_steps, update_database=False,
                    render=False, store_video=True, wandb_project=wandb_project,
                    save_model=False, greedy_sampling=greedy_sampling, n_step_td=1,
                    init_sac=False, init_rnd=False
                    )

                wandb.log(
                    {
                        'episode_eval': episode//eval_each, 
                        'eval_return': eval_returns.mean(),
                    }
                ) 

        return_array = np.array(returns)

        # Finish video recording
        if store_video:
            video.release()
            wandb.log(
                {'video': wandb.Video(video_filename, fps=4, format='mp4')}
            )

        # Close env
        if render:
            env.close()

        return return_array    

    def initialize(
        self, env, agents, database, n_step_td, max_episode_steps, 
        init_steps_sac, init_sac=True, init_rnd=True
        ):

        init_steps = init_steps_sac

        if init_rnd:
            rnd_module = agents[-1].rnd_module
            init_steps_rnd = rnd_module.pre_obs_norm_step
            init_steps = max(init_steps,  init_steps_rnd)

        step_counter = 0
        initialized = False

        while not initialized:

            state_buffer = deque(maxlen=max_episode_steps)
            action_buffer = deque(maxlen=max_episode_steps)
            reward_buffer = deque(maxlen=max_episode_steps)
            done_buffer = deque(maxlen=max_episode_steps)
            dist_buffer = deque(maxlen=max_episode_steps)
            
            step_counter_episode = 0
            episode_done = False
            state = env.reset()
            
            state_buffer.append(state['pixel'])

            while not episode_done:
                action = np.random.randint(agents[-1]._n_actions)
                dist = np.ones(agents[-1]._n_actions) / agents[-1]._n_actions
                next_state, reward, done, _ = env.step(action)

                if init_sac:
                    action_buffer.append(action)
                    reward_buffer.append(reward)
                    done_buffer.append(done)
                    dist_buffer.append(dist)                    

                state = next_state.copy()
                state_buffer.append(state['pixel'])
                
                step_counter_episode += 1
                step_counter += 1

                # Finish episode
                finished_episode = step_counter_episode >= max_episode_steps or done
                initialized = step_counter >= init_steps
                if finished_episode or initialized:
                    episode_done = True
                    if init_rnd:
                        # Update RND observation statistics
                        state_init_batch = np.stack(state_buffer).astype(
                            np.float)/255.
                        rnd_module.obs_rms.update(state_init_batch)
                        
                        # Update RND return statistics
                        int_rewards = agents[-1].rnd_module(state_init_batch)
                        int_pseudo_returns = [
                            rnd_module.discounted_reward.update(reward.item())
                            for reward in int_rewards]
                        int_pseudo_returns = np.stack(int_pseudo_returns)
                        rnd_module.reward_rms.update_from_moments(
                            np.mean(int_pseudo_returns),
                            np.std(int_pseudo_returns)**2,
                            len(int_pseudo_returns)
                        )

                    if init_sac: 
                        done_buffer[-1] = True
                        episode_trajectory = ExperienceFirstLevel(
                            state_buffer, action_buffer, reward_buffer, 
                            done_buffer, dist_buffer
                        )
                        database.append(episode_trajectory)

