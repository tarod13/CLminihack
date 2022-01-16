import collections
import numpy as np
import gym
import cv2


class AntPixelWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return AntPixelWrapper.separate_state(obs)

    @staticmethod
    def separate_state(obs):
        state = collections.OrderedDict()
        state['inner_state'] = obs['state'][2:-60] # Eliminate the xy coordinates (first 2 entries) and the 'lidar' maze observations
        outer_state = obs['pixels'] #.astype(np.float) / 255.0 -> not appropriate for memory
        outer_state = np.swapaxes(outer_state, 1, 2)
        state['outer_state'] = np.swapaxes(outer_state, 0, 1)
        state['first_level_obs'] = obs['state'][2:] #np.concatenate((obs['state'][2:-81],obs['state'][-60:])) #before: [2:-62]
        return state


class MinihackPixelWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, use_grayscale: bool = False, scale=0.125) -> None:
        super().__init__(env)
        self.use_grayscale = use_grayscale
        self.scale = scale

    def observation(self, obs):
        return self.transforms(obs)

    def transforms(self, obs):
        state = collections.OrderedDict()
        pixel = obs['pixel']
        state['original'] = pixel
        
        width = int(pixel.shape[1] * self.scale)
        height = int(pixel.shape[0] * self.scale)
        dims = (width, height)

        if self.use_grayscale:
            pixel = cv2.cvtColor(pixel, cv2.COLOR_RGB2GRAY)
        pixel = cv2.resize(pixel, dims, interpolation=cv2.INTER_AREA)
        if self.use_grayscale:
            pixel = np.expand_dims(pixel, axis=2)
        pixel = np.swapaxes(pixel, 1, 2)
        pixel = np.swapaxes(pixel, 0, 1)        
        state['pixel'] = pixel
        return state