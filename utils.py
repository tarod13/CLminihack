import numpy as np
import torch
import random
import time
import datetime
import yaml
import pickle

from buffers import ExperienceBuffer


def numpy2torch(np_array, device='cuda'):
    return torch.FloatTensor(np_array).to(device)

def updateNet(target, source, tau):    
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

def scale_action(a, min, max):
    return (0.5*(a+1.0)*(max-min) + min)

def vectorized_multinomial(prob_matrix):
    items = np.arange(prob_matrix.shape[1])
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0],1)
    k = (s < r).sum(axis=1)
    return items[k]

def one_hot_embedding(labels, num_classes, device='cuda'):
    y = torch.eye(num_classes).to(device) 
    return y[labels]

def set_seed(n_seed=0, device='cuda'):
    random.seed(n_seed)
    np.random.seed(n_seed)
    torch.manual_seed(n_seed)
    if device == 'cuda': torch.cuda.manual_seed(n_seed)

def is_float(x):
    return isinstance(x, float)

def is_tensor(x):
    return isinstance(x, torch.FloatTensor) or isinstance(x, torch.Tensor)

def time_stamp():
    time_in_seconds = time.time()
    stamp = datetime.datetime.fromtimestamp(time_in_seconds).strftime('%Y-%m-%d_%H-%M-%S')
    return stamp

def cat_state_task(observation):
    state = observation['state']
    task = observation['task']
    obs = np.concatenate((state, task))
    return obs

def load_env_model_pairs(file):
    yaml_file = open(file, 'r')
    try:
        env_model_pairs = yaml.load(yaml_file, Loader=yaml.FullLoader)['env_model_pairs']
        assert isinstance(env_model_pairs, dict)
    except:
        raise RuntimeError('Invalid file. It should be a dictionary with name "env_model_pairs"')
    return env_model_pairs

def load_database(n_parts, LOAD_PATH, DB_ID, buffer_size ,level):
    database = ExperienceBuffer(buffer_size, level)
    for i in range(0, n_parts):
        PATH = LOAD_PATH + DB_ID + '/SAC_training_level2_database_part_' + str(i) + '.p'
        database.buffer += pickle.load(open(PATH, 'rb'))
    return database

def separate_database(database):
    train_database = database
    train_database.sort()
    test_database = ExperienceBuffer(database._capacity, database._level)
    test_steps = [train_database.buffer.pop() for x in range(0,database._capacity//10)]
    test_database.buffer.extend(test_steps)
    return train_database, test_database

def temperature_search(
    q_values: torch.Tensor, 
    desired_entropy: float, 
    tolerance = 1e-2, 
    max_iter = 100,
    c_plus: float = 100,
    c_minus: float = 0.01,
    verbose: bool = False) -> torch.Tensor:
    ''' 
    Find the temperatures that ensure that the distributions 
        
            p(a|s) = exp(q(s,a)/temperature(s))/Z(s) 
    
    have the desired entropy, with a margin of error given the
    tolerance. It is supposed that the last dimension correspond
    to the possible discrete actions.
    '''
    # if len(q_values.shape) != 2:
    #     raise RuntimeError("Q-values should be matrices, i.e., 2d tensors")

    n_actions = q_values.shape[-1]

    log_alpha_plus = torch.log(
        c_plus * q_values.abs().sum(-1, keepdim=True) + 1e-10)
    
    log_alpha_minus = torch.log(
        c_minus * q_values.abs().min(-1, keepdim=True)[0] + 1e-10)

    worst_entropy_difference = np.log(n_actions)
    n_iter = 0
    while worst_entropy_difference > tolerance and n_iter < max_iter:
        # Set log-temperature as the average of the plus and minus estimates
        log_alpha = 0.5 * (log_alpha_plus + log_alpha_minus)
        alpha = torch.exp(log_alpha)
        
        # Calculate entropies
        q_normalized = q_values / alpha
        q = q_normalized - q_normalized.max(-1, keepdim=True)[0]
        distribution_ = torch.exp(q)+1e-10
        distribution_ /= distribution_.sum(-1, keepdim=True)
        entropy = -(
            distribution_ * torch.log(distribution_)
        ).sum(-1, keepdim=True)

        # Update temperature estimates depending on whether the entropy
        # difference is positive or negative
        entropy_difference = entropy - desired_entropy
        log_alpha_plus = torch.where(
            entropy_difference > 0, log_alpha, log_alpha_plus)
        log_alpha_minus = torch.where(
            entropy_difference <= 0, log_alpha, log_alpha_minus)

        worst_entropy_difference = entropy_difference.abs().max()
        n_iter += 1

        if verbose:
            print(
                f'Max temp: {np.exp(log_alpha_plus.max().item()):.3e}, ' +  
                f'Min temp: {np.exp(log_alpha_minus.min()):.3e},' + 
                f'Max entropy: {entropy.max():.3f}, ' +  
                f'Min entropy: {entropy.min():.3f},' + 
                f'Worst entropy delta: {worst_entropy_difference:.3f}'
            )
    
    return 0.5 * (log_alpha_plus + log_alpha_minus), n_iter