import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import Adam

from policy_nets import softmax_policy_Net, vision_softmax_policy_Net
from vision_nets import vision_Net
from custom_layers import Linear_noisy
from net_utils import *


class concept_Net(nn.Module):
    def __init__(self, s_dim, n_concepts, noisy=False):
        super().__init__()

        self.classifier = softmax_policy_Net(s_dim, n_concepts, noisy=noisy)
        self._n_concepts = n_concepts

    def forward(self, s):
        PS_s, log_PS_s = self.classifier(s)
        return PS_s, log_PS_s 


class SA_concept_Net(nn.Module):
    def __init__(self, s_dim, n_state_concepts, a_dim, n_action_concepts, 
        noisy=False, init_log_alpha=1.0):
        super().__init__()

        self.state_net = concept_Net(s_dim, n_state_concepts, noisy)
        self.action_net = concept_Net(a_dim, n_action_concepts, noisy)
        self.log_alpha = Parameter(torch.Tensor(1))
        nn.init.constant_(self.log_alpha, init_log_alpha)
        self.alpha_optimizer = Adam([self.log_alpha], lr=3e-4)

    def forward(self, s, a):
        PS_s, log_PS_s = self.state_net(s)
        PA_a, log_PA_a = self.action_net(a)
        return PS_s, log_PS_s, PA_a, log_PA_a 


class visual_S_concept_Net(nn.Module):
    def __init__(self, s_dim, latent_dim, n_concepts, noisy=False, lr=1e-4):
        super().__init__()

        self.s_dim = s_dim
        self.classifier = vision_softmax_policy_Net(s_dim, latent_dim, n_concepts, noisy, lr)
        self._n_concepts = n_concepts

    def forward(self, inner_state, outer_state):
        PS_s, log_PS_s = self.classifier(None, outer_state)
        return PS_s, log_PS_s 


# Continuous concepts
#---------------------------------------------------------------------------

class state_encoder_Net(nn.Module):
    def __init__(self, s_dim, c_dim=10, noisy=False, lr=3e-4, latent_dim=32):
        super().__init__()
        
        self.s_dim = s_dim   
        self.c_dim = c_dim 

        if noisy:
            layer = Linear_noisy
        else:
            layer = nn.Linear

        self.last_layer = layer(latent_dim, c_dim)
        self.pipe = nn.Sequential(
            layer(s_dim, latent_dim),
            nn.ReLU(),
            layer(latent_dim, latent_dim),
            nn.ReLU(),
            self.last_layer            
        )        
        
        if not noisy:
            self.pipe.apply(weights_init_rnd)
            torch.nn.init.orthogonal_(self.last_layer.weight, 0.01)
            self.last_layer.bias.data.zero_()
        else:
            torch.nn.init.orthogonal_(self.last_layer.mean_weight, 0.01)
            self.last_layer.mean_bias.data.zero_()
            
        self.optimizer = Adam(self.parameters(), lr=lr)
        
    def forward(self, s):    
        return self.pipe(s)


class visual_continuous_concept_Net(state_encoder_Net):
    def __init__(self, s_dim, vision_latent_dim, c_dim=10, latent_dim=32, noisy=True, lr=1e-4):
        super().__init__(s_dim + latent_dim, c_dim, noisy, lr, latent_dim)        
        self.vision_net = vision_Net(latent_dim=vision_latent_dim, noisy=noisy)
        self.optimizer = Adam(self.parameters(), lr=lr)
        
    def forward(self, inner_state, outer_state):    
        features = self.vision_net(outer_state)
        state = torch.cat([inner_state, features], dim=1)
        code = self.pipe(state) 
        return code