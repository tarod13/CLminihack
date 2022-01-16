from collections import namedtuple, deque
import itertools
import numpy as np
import torch
import random

ExperienceFirstLevel = namedtuple(
    'ExperienceFirstLevel', 
    field_names=['pixels', 'actions', 'rewards', 'dones', 'dists'])


def sample_from_deque(d, idx0, idxf):
    return list(itertools.islice(d, idx0, idxf))


class EpisodeExperienceBuffer:
    def __init__(self, capacity, level=1):
        self.buffer = deque(maxlen=capacity)
        self._level = level
        self._capacity = capacity

        assert level in [1], 'Invalid level. Must be 1.'

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)
    
    def sort(self):
        episodes = [self.buffer.pop() for x in range(0,self.__len__())]
        random.shuffle(episodes)
        self.buffer.extend(episodes)

    def sample(
        self, batch_size:int, random_sample:bool=True, 
        n_step:int=1, to_torch=True, dev_name='cuda'
        ) -> list:
        '''
        Return a random sample of the buffer. If not random_sample,
        then the whole buffer is returned.
        '''
        if random_sample:
            epsd_indices = np.random.choice(
                len(self.buffer), batch_size, replace=True
            )
        else:
            epsd_indices = range(0, len(self.buffer))

        if self._level == 1:
            sampled_pixels = []
            sampled_actions = []
            sampled_rewards = []
            sampled_dones = []
            sampled_dists = []

            for epsd_idx in epsd_indices:
                epsd_pixels, epsd_actions, epsd_rewards, \
                    epsd_dones, epsd_dists = self.buffer[epsd_idx]

                n_steps_available = len(epsd_actions) - n_step
                is_long_episode = n_steps_available > 0
                if is_long_episode:
                    step_idx_init = np.random.randint(n_steps_available)
                    step_idx_fin = step_idx_init + n_step
                else:
                    step_idx_init = 0
                    step_idx_fin = len(epsd_actions)

                pixels = sample_from_deque(
                    epsd_pixels, step_idx_init, step_idx_fin+1)
                actions = sample_from_deque(
                    epsd_actions, step_idx_init, step_idx_fin)
                rewards = sample_from_deque(
                    epsd_rewards, step_idx_init, step_idx_fin)
                dones = sample_from_deque(
                    epsd_dones, step_idx_init, step_idx_fin)
                dists = sample_from_deque(
                    epsd_dists, step_idx_init, step_idx_fin)

                if not is_long_episode:
                    steps_missing = np.abs(n_steps_available)
                    pixels = pixels + steps_missing*[pixels[-1]]
                    actions = actions + steps_missing*[actions[-1]]
                    rewards = rewards + steps_missing*[0.0]
                    dones = dones + steps_missing*[True]
                    dists = dists + steps_missing*[dists[-1]]
                    
                sampled_pixels.append(np.stack(pixels).astype(np.uint8))
                # if len(actions.shape) == 0:
                #     actions = np.array([actions])
                #     rewards = np.array([actions])
                sampled_actions.append(np.stack(actions).astype(np.uint8))
                sampled_rewards.append(np.stack(rewards).astype(np.float32))
                sampled_dones.append(np.stack(dones).astype(np.uint8))
                sampled_dists.append(np.stack(dists).astype(np.float32))

            if to_torch:
                device = torch.device(dev_name)
                sampled_pixels = np.stack(sampled_pixels).astype(np.float32)/255.
                pixels_th = torch.FloatTensor(sampled_pixels).to(device)
                rewards_th = torch.FloatTensor(np.stack(sampled_rewards)).to(device)
                dones_th = torch.ByteTensor(np.stack(sampled_dones)).float().to(device)
                dists_th = torch.FloatTensor(np.stack(sampled_dists)).to(device)
                return pixels_th, np.stack(sampled_actions).astype('int'), \
                    rewards_th, dones_th, dists_th
            
            else:
                return np.stack(sampled_pixels), np.stack(sampled_actions), \
                    np.stack(sampled_rewards), np.stack(sampled_dones), \
                    np.stack(sampled_dists)
