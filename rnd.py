import torch
import torch.nn as nn
import numpy as np
import torch
from torch.optim import Adam
from vision_nets import vision_Net
from net_utils import *
from utils import numpy2torch as np2torch

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0, keepdims=True)
        batch_var = np.var(x, axis=0, keepdims=True)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class RND_Net(nn.Module):
    def __init__(self, input_shape=(1,1,42,158), latent_dim=128, out_dim=512, lr=1e-4):
        super().__init__()  
        self.vision_net = vision_Net(
            latent_dim=latent_dim, 
            input_channels=input_shape[1], 
            height=input_shape[2], 
            width=input_shape[3],
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
    def __init__(self, input_shape, latent_dim, out_dim=512):
        super().__init__(input_shape, latent_dim, out_dim)  
        self.optimizer = None
        for param in self.parameters():
            param.requires_grad = False


class RND_Module(nn.Module):
    def __init__(
        self, input_shape=(1,1,42,158), latent_dim=1024, out_dim=128, lr=1e-4, 
        obs_norm_step=50, int_gamma=0.99
        ):

        super().__init__() 
        self.latent_dim = latent_dim
        self.out_dim = out_dim

        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=input_shape)
        self.pre_obs_norm_step = obs_norm_step
        self.discounted_reward = RewardForwardFilter(int_gamma)
        
        self.target = RND_targetNet(input_shape, latent_dim, out_dim)
        self.predictor = RND_Net(input_shape, latent_dim, out_dim, lr=lr)
        self.predictor_frozen = RND_targetNet(input_shape, latent_dim, out_dim)
        
        updateNet(self.predictor_frozen, self.predictor, 1.0)

    def normalize_obs(self, pixels_np):
        pixels_mean = self.obs_rms.mean
        pixels_var = self.obs_rms.mean
        pixels_norm = (pixels_np - pixels_mean) / ((pixels_var)**0.5 + 1e-10)
        pixels = np2torch(pixels_norm.clip(-5, 5)).to(device)
        return pixels

    def forward(self, pixels):
        pixels_normalized = self.normalize_obs(pixels)
        target = self.target(pixels_normalized)
        prediction = self.predictor(pixels_normalized)
        error = ((prediction - target.detach())**2).sum(1, keepdim=True)
        return error

    def calc_novelty(self, pixels):
        with torch.no_grad():
            pixels_normalized = self.normalize_obs(
                pixels.detach().cpu().numpy())
            target = self.target(pixels_normalized)
            prediction = self.predictor(pixels_normalized)
            prediction_0 = self.predictor_frozen(pixels_normalized)

            error = ((prediction - target)**2).sum(1, keepdim=True)
            error_0 = ((prediction_0 - target)**2).sum(1, keepdim=True)

            log_novelty = (
                torch.log(error + 1e-10) 
                - torch.log(error_0 + 1e-10)
            )
            return log_novelty, error/(self.reward_rms.var**0.5)