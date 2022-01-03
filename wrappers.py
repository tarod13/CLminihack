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


class TransposeChannelWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return TransposeChannelWrapper.transpose(obs)

    @staticmethod
    def transpose(obs):
        state = collections.OrderedDict()
        pixel = obs['pixel']
        state['original'] = pixel
        
        scale = 0.25        
        width = int(pixel.shape[1] * scale)
        height = int(pixel.shape[0] * scale)
        dims = (width, height)

        pixel = cv2.resize(pixel, dims, interpolation=cv2.INTER_AREA)
        pixel = np.swapaxes(pixel, 1, 2)
        pixel = np.swapaxes(pixel, 0, 1)        
        state['pixel'] = pixel
        return state