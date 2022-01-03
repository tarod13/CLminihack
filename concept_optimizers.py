import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from policy_optimizers import Optimizer
from utils import one_hot_embedding

class SA_ConceptOptimizer(Optimizer):
    def __init__(self, batch_size=32, clip_value=1.0, 
                lr=3e-4
                ):
        super().__init__()
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.lr = lr
        
    def optimize(self, agent, database, IAs_T_min=1.36): 
        if database.__len__() < self.batch_size:
            return None

        # Set alias for concept network with state and action classifier
        concept_net = agent.concept_architecture

        # Sample batch
        if agent.first_level_agent._type == 'multitask':
            observations, actions, rewards, dones, next_observations = \
                database.sample(self.batch_size)
            
            PS_sT, log_PS_sT, PA_sT, log_PA_sT = agent(observations)
           
            NAST = torch.einsum('ij,ik->jk', PS_sT, PA_sT).detach() + 1e-8
            
            PS_T = NAST.sum(1) / NAST.sum(1).sum(0)
            PA_ST = NAST / NAST.sum(1, keepdim=True)
            PA_T = NAST.sum(0) / NAST.sum(0).sum(0)

            log_PS_T = torch.log(PS_T)
            log_PA_T = torch.log(PA_T)
            log_PA_ST = torch.log(PA_ST)            
            
            HS_T = -(PS_sT * log_PS_T.unsqueeze(0)).sum(1).mean()
            HS_sT = -(PS_sT * log_PS_sT).sum(1).mean()
            ISs_T = HS_T - HS_sT
            
            HA_T = -(PA_sT * log_PA_T.unsqueeze(0)).sum(1).mean()
            HA_sT = -(PA_sT * log_PA_sT).sum(1).mean()
            HA_ST_state = -((log_PA_ST.unsqueeze(0) * PA_sT.unsqueeze(1).detach()).sum(2) * PS_sT).sum(1).mean()  
            HA_ST_action = -((log_PA_ST.unsqueeze(0) * PA_sT.unsqueeze(1)).sum(2) * PS_sT.detach()).sum(1).mean()  
            
            IAS_T = HA_T.detach() - HA_ST_state
            state_classifier_loss = -IAS_T

            # Optimize state classifier
            concept_net.state_net.classifier.optimizer.zero_grad()
            state_classifier_loss.backward()
            clip_grad_norm_(concept_net.state_net.classifier.parameters(), self.clip_value)
            concept_net.state_net.classifier.optimizer.step()

            IAs_T = HA_T - HA_sT          
            IAS_T_action = HA_T - HA_ST_action
            IAs_ST = IAs_T - IAS_T_action
            alpha = concept_net.log_alpha.exp().item()
            action_classifier_loss = IAs_ST - alpha * IAs_T
            
            # Optimize action classifier
            concept_net.action_net.classifier.optimizer.zero_grad()
            action_classifier_loss.backward()
            clip_grad_norm_(concept_net.action_net.classifier.parameters(), self.clip_value)
            concept_net.action_net.classifier.optimizer.step() 

            alpha_loss = concept_net.log_alpha * (IAs_T - IAs_T_min).detach() 
            
            # Optimize temperature parameter
            concept_net.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            clip_grad_norm_([concept_net.log_alpha], self.clip_value)
            concept_net.alpha_optimizer.step()


            metrics = {
                'HS_T': HS_T.item(),
                'HS_sT': HS_T.item(),
                'HA_T': HS_T.item(),
                'HA_sT': HS_T.item(),
                'HA_ST': HS_T.item(),
                'ISs_T': ISs_T.item(),
                'IAs_T': IAs_T.item(),
                'IAS_T': IAS_T.item(),
                'IAs_ST': IAs_ST.item(),
                'state_loss': state_classifier_loss.item(),
                'action_loss': action_classifier_loss.item(),
                'alpha_loss': alpha_loss.item(),
                'alpha': alpha,
            }
            
            return metrics


class S_ConceptOptimizer(Optimizer):
    def __init__(self, batch_size=64, beta=0.0, eta=0.0, n_batches_estimation=2,
        update_rate=0.05, consider_task=True, detach_logs=True, clip_value=1.0):
        super().__init__()
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.beta = beta
        self.eta = eta
        self.n_batches_estimation = n_batches_estimation
        self.update_rate = update_rate
        self.consider_task = consider_task
        self.detach_logs = detach_logs
        self.PAST = None
        self.PAS = None
        self.PAST_test = None
        self.PAS_test = None
        
    def optimize(self, agent, database, n_actions, n_tasks, initialization=False, train=True): 
        if database.__len__() < self.batch_size:
            return None

        if train:
            PAST = self.PAST
            PAS = self.PAS
        else:
            PAST = self.PAST_test
            PAS = self.PAS_test

        if (self.n_batches_estimation > 1) or initialization:
            if not initialization:
                n_batches = self.n_batches_estimation
            else:
                n_batches = 200
                print("Initializing estimation")
            # Estimate visits
            NAST = None

            with torch.no_grad():
                for batch in range(0, n_batches):
                    # Sample batch        
                    inner_states, outer_states, actions, rewards, dones, \
                        next_inner_states, next_outer_states, tasks = \
                        database.sample(self.batch_size)
                    
                    PS_s, log_PS_s = agent(inner_states, outer_states)
                    A_one_hot = one_hot_embedding(actions, n_actions)
                    T_one_hot = one_hot_embedding(tasks, n_tasks)

                    NAS_batch = PS_s.unsqueeze(2) * A_one_hot.unsqueeze(1)
                    NAST_batch = torch.einsum('ijk,ih->hjk', NAS_batch, T_one_hot).detach()

                    if NAST is None:
                        NAST = NAST_batch
                    else:
                        NAST = NAST + NAST_batch
            
            NAST += 1e-8

        # Sample batch        
        inner_states, outer_states, actions, rewards, dones, \
            next_inner_states, next_outer_states, tasks = \
            database.sample(self.batch_size)
        
        PS_s, log_PS_s = agent(inner_states, outer_states)
        A_one_hot = one_hot_embedding(actions, n_actions)
        T_one_hot = one_hot_embedding(tasks, n_tasks)
        
        if (self.n_batches_estimation == 1) and not initialization:
            if self.detach_logs:
                PAS_batch = PS_s.detach().unsqueeze(2) * A_one_hot.unsqueeze(1)
            else:
                PAS_batch = PS_s.unsqueeze(2) * A_one_hot.unsqueeze(1)
            NAST = torch.einsum('ijk,ih->hjk', PAS_batch, T_one_hot) + 1e-20
            NAS = PAS_batch.sum(0) + 1e-20
        else:
            NAS = NAST.sum(0)
        
        PAST_batch = NAST / NAST.sum()
        PAS_batch = NAS / NAS.sum()

        if PAST is None:
            PAST = PAST_batch
        else:
            PAST = PAST * (1.-self.update_rate) + PAST_batch * self.update_rate

        if PAS is None:
            PAS = PAS_batch
        else:
            PAS = PAS * (1.-self.update_rate) + PAS_batch * self.update_rate

        # PAST = NAST / NAST.sum()
        PT = PAST.sum(2).sum(1)
        PT = PT / PT.sum()
        PST = PAST.sum(2)
        PS_T = PST / PST.sum(1, keepdim=True)
        PA_ST = PAST / PAST.sum(2, keepdim=True)
        PAT = PAST.sum(1)
        PA_T = PAT / PAT.sum(1, keepdim=True)
        PAS_T = PAST / PAST.sum(2).sum(1)
        PA = PAS.sum(0)
        PS = PAS.sum(1)
        PA_S = PAS / PAS.sum(1, keepdim=True)

        log_PS_T = torch.log(PS_T)
        log_PA_T = torch.log(PA_T)
        log_PA_ST = torch.log(PA_ST) 
        log_PA = torch.log(PA)
        log_PS = torch.log(PS)
        log_PA_S = torch.log(PA_S)                
        
        T_one_hot_dist = (T_one_hot + 1e-20) / (T_one_hot + 1e-20).sum(0, keepdim=True)
        PS_sgT = PS_s.detach().unsqueeze(1) * T_one_hot_dist.unsqueeze(2)
        HS_gT_samp = torch.einsum('ihj,hj->ih', PS_sgT, -log_PS_T).sum(0)
        HS_gT = torch.einsum('ij,ij->i', PS_s, -log_PS_T[tasks,:]).mean()
        HS_s = -(PS_s * log_PS_s).sum(1).mean()
        ISs_gT = HS_gT_samp - HS_s
        ISs_T = HS_gT - HS_s
        
        HA_gT = -(PA_T * log_PA_T).sum(1)
        HA_T = (PT * HA_gT).sum()
        HA_sT = 0.0*np.log(n_actions)
        assert log_PA_ST[tasks,:,actions].shape == PS_s.shape, 'Problems with dimensions (T)'
        HA_ST = -(PS_s * log_PA_ST[tasks,:,actions]).sum(1).mean()
        HA_SgT = -(PAS_T * log_PA_ST).sum((1,2))  

        HS = -(PS_s.mean(0) * log_PS).sum()
        ISs = HS - HS_s
        IST = HS - HS_gT

        HA_s = 0.0*np.log(n_actions)
        HA = -(PA * log_PA).sum()
        assert log_PA_S[:,actions].T.shape == PS_s.shape, 'Problems with dimensions (NT)'
        HA_S = -(PS_s * log_PA_S[:,actions].T).sum(1).mean()
        
        IAs_gT = HA_gT - HA_sT   
        IAS_gT = HA_gT - HA_SgT
        IAs_SgT = IAs_gT - IAS_gT

        IAs_T = (PT * IAs_gT).sum()
        IAS_T = HA_T - HA_ST
        IAs_ST = IAs_T - IAS_T         
        
        IAs = HA - HA_s
        IAS = HA - HA_S
        IAs_S = IAs - IAS

        n_concepts = PS_s.shape[1]
        H_max = np.log(n_concepts)
        if self.consider_task:
            classifier_loss = IAs_ST + self.beta * ISs_T + self.eta * IST
        else:
            classifier_loss = IAs_S + self.beta * ISs                

        if train:
            agent.classifier.optimizer.zero_grad()
            classifier_loss.backward()
            clip_grad_norm_(agent.classifier.parameters(), self.clip_value)
            agent.classifier.optimizer.step()

        if not self.detach_logs:
            PAST = PAST.detach()
            PAS = PAS.detach()
        
        if train:
            self.PAST = PAST
            self.PAS = PAS
        else:
            self.PAST_test = PAST 
            self.PAS_test = PAS 

        label = ''
        if not train:
            label += '_test'

        joint_metrics = {
            'HA'+label: HA.item(),
            'HA_S'+label: HA_S.item(),
            'HS'+label: HS.item(),
            'HS_T'+label: HS_gT.mean().item(),
            'HS_s'+label: HS_s.item(),
            'HA_T'+label: HA_T.item(),
            'HA_sT'+label: HA_sT,
            'HA_ST'+label: HA_ST.item(),
            'IST'+label: IST.item(),
            'ISs'+label: ISs.item(),
            'ISs_T'+label: ISs_T.item(),
            'IAs'+label: IAs.item(),
            'IAS'+label: IAS.item(),
            'IAs_S'+label: IAs_S.item(),
            'IAs_T'+label: IAs_T.item(),
            'IAS_T'+label: IAS_T.item(),
            'IAs_ST'+label: IAs_ST.item(),
            'loss'+label: classifier_loss.item(),
        }

        metrics_per_task = {}
        for task in range(0, n_tasks):
            metrics_per_task['HS_T'+str(task)+label] = HS_gT_samp[task].item()
            metrics_per_task['HA_T'+str(task)+label] = HA_gT[task].item()
            metrics_per_task['HA_ST'+str(task)+label] = HA_SgT[task].item()
            metrics_per_task['ISs_T'+str(task)+label] = ISs_gT[task].item()
            metrics_per_task['IAs_T'+str(task)+label] = IAs_gT[task].item()
            metrics_per_task['IAS_T'+str(task)+label] = IAS_gT[task].item()
            metrics_per_task['IAs_ST'+str(task)+label] = IAs_SgT[task].item()

        metrics = {**joint_metrics, **metrics_per_task}
        
        return metrics



class ConceptOptimizer(Optimizer):
    def __init__(self, batch_size=64, beta=0.0, n_batches_estimation=2, update_rate=0.05, clip_value=1.0):
        super().__init__()
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.beta = beta
        self.n_batches_estimation = n_batches_estimation
        self.update_rate = update_rate
        self.PAT = None
        
    def optimize(self, agent, conceptual_policy, database, n_actions, n_tasks): 
        if database.__len__() < self.batch_size:
            return None

        # # Estimate visits
        # NAST = None

        # with torch.no_grad():
        #     for batch in range(0, self.n_batches_estimation*8):
        #         # Sample batch        
        #         inner_states, outer_states, actions, rewards, dones, \
        #             next_inner_states, next_outer_states, tasks = \
        #             database.sample(self.batch_size)
                
        #         PS_s, log_PS_s = agent(inner_states, outer_states)
        #         A_one_hot = one_hot_embedding(actions, n_actions)
        #         T_one_hot = one_hot_embedding(tasks, n_tasks)

        #         PAS = PS_s.unsqueeze(2) * A_one_hot.unsqueeze(1)
        #         NAST_batch = torch.einsum('ijk,ih->hjk', PAS, T_one_hot).detach() + 1e-8

        #         if NAST is None:
        #             NAST = NAST_batch
        #         else:
        #             NAST = NAST + NAST_batch

        # Sample batch        
        inner_states, outer_states, actions, rewards, dones, \
            next_inner_states, next_outer_states, tasks = \
            database.sample(self.batch_size)
        
        z = agent(inner_states, outer_states)
        PA_zT, log_PA_zT = conceptual_policy(z)
        PA_zgT = PA_zgT[np.arange(0,self.batch_size),tasks,:]
        log_PA_zgT = log_PA_zgT[np.arange(0,self.batch_size),tasks,:]

        A_one_hot = one_hot_embedding(actions, n_actions)
        T_one_hot = one_hot_embedding(tasks, n_tasks)

        NAT = torch.einsum('ik,ih->hk', A_one_hot, T_one_hot).detach() + 1e-8
        PAT_batch = NAT / NAT.sum()

        if self.PAT is None:
            self.PAT = PAT_batch
        else:
            self.PAT = self.PAT * (1.-self.update_rate) + PAT_batch * self.update_rate

        # PAST = NAST / NAST.sum()
        PT = self.PAT.sum(1)
        PA_T = self.PAT / PT.view(-1,1)        
        log_PA_T = torch.log(PA_T)
        
        HA_gT = -(PA_T * log_PA_T).sum(1)
        HA_T = (PT * HA_gT).sum()
        HA_sT = 0.03*np.log(n_actions)
        HA_zgT = -((PA_zT * T_one_hot.unsqueeze(1)) * log_PA_zT).sum(2).mean(1)
        HA_zT = -(PA_zgT * log_PA_zgT).sum(1).mean()
        HA_zT_nosamp = (PT * HA_zgT).sum()
                
        IAs_gT = HA_gT - HA_sT   
        IAz_gT = HA_gT - HA_zgT
        IAs_zgT = IAs_gT - IAz_gT

        IAs_T = (PT * IAs_gT).sum()
        IAz_T = HA_T - HA_zT
        IAs_ST = IAs_T - IAz_T
         
        
        n_concepts = PS_s.shape[1]
        H_max = np.log(n_concepts)
        classifier_loss = IAs_ST #+ self.beta * ISs_T            

        agent.classifier.optimizer.zero_grad()
        classifier_loss.backward(retain_graph=True)
        clip_grad_norm_(agent.classifier.parameters(), self.clip_value)
        agent.classifier.optimizer.step()

        conceptual_policy.classifier.optimizer.zero_grad()
        classifier_loss.backward()
        clip_grad_norm_(conceptual_policy.classifier.parameters(), self.clip_value)
        conceptual_policy.classifier.optimizer.step()

        joint_metrics = {
            'HA_T': HA_T.item(),
            'HA_sT': HA_sT,
            'HA_ST': HA_zT.item(),
            'HA_ST_nosamp': HA_zT_nosamp.item(),
            'IAs_T': IAs_T.item(),
            'IAS_T': IAz_T.item(),
            'IAs_ST': IAs_zT.item(),
            'loss': classifier_loss.item(),
        }

        metrics_per_task = {}
        for task in range(0, n_tasks):
            metrics_per_task['HA_T'+str(task)] = HA_gT[task].item()
            metrics_per_task['HA_ST'+str(task)] = HA_zgT[task].item()
            metrics_per_task['IAs_T'+str(task)] = IAs_gT[task].item()
            metrics_per_task['IAS_T'+str(task)] = IAz_gT[task].item()
            metrics_per_task['IAs_ST'+str(task)] = IAs_zgT[task].item()

        metrics = {**joint_metrics, **metrics_per_task}
        
        return metrics