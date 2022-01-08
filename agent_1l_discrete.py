import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from actor_critic_nets import discrete_vision_actor_critic_Net
from rnd import RND_Module
from net_utils import freeze
from utils import numpy2torch as np2torch
from utils import time_stamp


def create_second_level_agent(
    n_actions=8, latent_dim=256, n_heads=8, init_log_alpha=0.0, 
    noop_action=False, device='cuda', noisy=True, parallel=True, 
    lr=1e-4, lr_alpha=1e-4, lr_actor=1e-4, rnd_out_dim=128):
    
    second_level_architecture = discrete_vision_actor_critic_Net(
        n_actions+int(noop_action), latent_dim, n_heads, 
        init_log_alpha, parallel, lr, lr_alpha, lr_actor
    )

    rnd_module = RND_Module(out_dim=rnd_out_dim)

    second_level_agent = Second_Level_Agent(
        n_actions, second_level_architecture, rnd_module, noop_action
    ).to(device)
    return second_level_agent


class Second_Level_Agent(nn.Module):
    def __init__(
        self, 
        n_actions: int, 
        second_level_architecture: discrete_vision_actor_critic_Net, 
        rnd_module: RND_Module, 
        noop_action: bool
        ):  
        super().__init__()    
        
        self.second_level_architecture = second_level_architecture
        self.rnd_module = rnd_module
        
        self._n_actions = n_actions + int(noop_action)
        self._noop = noop_action
        self._id = time_stamp()
    
    def forward(self, states):
        pass 

    def calc_novelty(self, pixels):
        return self.rnd_module.calc_novelty(pixels)
    
    def sample_action(self, state, explore=True):
        pixel = self.observe_second_level_state(state)
        with torch.no_grad():
            action, dist = self.second_level_architecture.sample_action(
                pixel, explore=explore)            
            return action, dist

    def observe_second_level_state(self, state):
        pixel_np = state['pixel'].astype(np.float)/255.        
        pixel = np2torch(pixel_np)        
        return pixel

    def train_rnd_module(self, pixels_np):
        pixels_mean = self.rnd_module.obs_rms.mean
        pixels_var = self.rnd_module.obs_rms.mean
        pixels_norm = (pixels_np - pixels_mean) / ((pixels_var)**0.5 + 1e-10)
        pixels = np2torch(pixels_norm.clip(-5, 5))
        rnd_loss = self.rnd_module(pixels).mean()
        self.rnd_module.predictor.optimizer.zero_grad()
        rnd_loss.backward()
        clip_grad_norm_(self.rnd_module.predictor.parameters(), 1.0)
        self.rnd_module.predictor.optimizer.step()
        return rnd_loss.item()
    
    def save(self, save_path, best=False):
        if best:
            model_path = save_path + 'best_agent_2l_' + self._id
        else:
            model_path = save_path + 'last_agent_2l_' + self._id
        torch.save(self.state_dict(), model_path)
    
    def load(self, load_directory_path, model_id, device='cuda'):
        dev = torch.device(device)
        self.load_state_dict(
            torch.load(
                load_directory_path + 'agent_2l_' + model_id, map_location=dev
            )
        )

    def get_id(self):
        return self._id


if __name__ == "__main__":
    agent = create_second_level_agent()
    print("Successful second level agent creation")