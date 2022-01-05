import torch
import torch.nn as nn
from torch.optim import Adam

from vision_nets import vision_Net
from net_utils import *

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class RND_Net(nn.Module):
    def __init__(self, latent_dim, out_dim=512, lr=1e-4):
        super().__init__()  
        self.vision_net = vision_Net(
            latent_dim=latent_dim, 
            noisy=False)
        self.linear_pipe = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, out_dim)  
        )    
        
        self.apply(weights_init_rnd)
        self.optimizer = Adam(self.parameters(), lr=lr)
   
    def forward(self, pixels):
        features = self.vision_net(pixels)
        return self.linear_pipe(features)


class RND_targetNet(RND_Net):
    def __init__(self, latent_dim, out_dim=512):
        super().__init__(latent_dim, out_dim)  
        self.optimizer = None
        for param in self.parameters():
            param.requires_grad = False


class RND_Module(nn.Module):
    def __init__(self, latent_dim=1024, out_dim=128, lr=1e-4):
        super().__init__() 
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        
        self.target = RND_targetNet(latent_dim, out_dim)
        self.predictor = RND_Net(latent_dim, out_dim, lr=lr)
        self.predictor_frozen = RND_targetNet(latent_dim, out_dim)
        
        updateNet(self.predictor_frozen, self.predictor, 1.0)

    def forward(self, pixels):
        target = self.target(pixels)
        prediction = self.predictor(pixels)
        error = ((prediction - target.detach())**2).sum(1, keepdim=True)
        return error

    def calc_novelty(self, pixels):
        with torch.no_grad():
            target = self.target(pixels)
            prediction = self.predictor(pixels)
            prediction_0 = self.predictor_frozen(pixels)

            error = ((prediction - target)**2).sum(1, keepdim=True)
            error_0 = ((prediction_0 - target)**2).sum(1, keepdim=True)

            log_novelty = torch.log(error + 1e-10) - torch.log(error_0 + 1e-10)
            return log_novelty, error, error_0