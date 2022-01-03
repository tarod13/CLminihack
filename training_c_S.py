import numpy as np

import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper
from agent_c_S import Conceptual_Agent
from concept_optimizers import S_ConceptOptimizer
from buffers import ExperienceBuffer
from trainers import Second_Level_Trainer as Trainer
from wrappers import AntPixelWrapper
from utils import load_database, separate_database

import argparse
import pickle
import os
import wandb


LOAD_PATH = '/home/researcher/Diego/Concept_Learning_minimal/saved_data/'
SAVE_PATH = '/home/researcher/Diego/Concept_Learning_minimal/saved_models/concept_models/'


DEFAULT_BUFFER_SIZE = 400000
DEFAULT_BATCH_SIZE = 2048
DEFAULT_BETA_REGULARIZATION = 0
DEFAULT_ETA_REGULARIZATION = 0
DEFAULT_CONSIDER_TASK = False
DEFAULT_DB_ID = '2021-02-19_12-52-25' #'2021-02-15_03-26-57' #'2021-02-14_16-42-35' # '2021-02-10_19-36-12' #'2021-02-10_00-42-26' #'2021-01-28_17-38-37'
DEFAULT_ID = None #'2021-02-16_18-29-26_v20' # None #'2021-02-16_12-50-45_v20' # None # '2021-02-15_12-57-00_v20'  #'2021-01-29_21-16-46'
DEFAULT_INNER_STATE_DIM = 0
DEFAULT_LR = 1e-4
DEFAULT_N_PARTS = 40
DEFAULT_N_STEPS = 100000
DEFAULT_N_TASKS = 4
DEFAULT_N_CONCEPTS = 20
DEFAULT_N_ACTIONS = 4
DEFAULT_N_BATCHES = 1
DEFAULT_N_SAVES = 100
DEFAULT_NOISY = False
DEFAULT_UPDATE_RATE = 2e-1
DEFAULT_VISION_LATENT_DIM = 64
DEFAULT_DETACH_LOGS = True



if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", default=DEFAULT_BETA_REGULARIZATION, help="Regularization level, default=" + str(DEFAULT_BETA_REGULARIZATION))
    parser.add_argument("--eta", default=DEFAULT_ETA_REGULARIZATION, help="Regularization level, default=" + str(DEFAULT_ETA_REGULARIZATION))
    parser.add_argument("--cpu", action="store_true", help="Disable cuda")
    parser.add_argument("--n_steps", default=DEFAULT_N_STEPS, help="Number of SGD steps taken, default=" + str(DEFAULT_N_STEPS))
    parser.add_argument("--buffer_size", default=DEFAULT_BUFFER_SIZE, help="Size of replay buffer, default=" + str(DEFAULT_BUFFER_SIZE))
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, help="Size of batch used in SGD, default=" + str(DEFAULT_BATCH_SIZE))
    parser.add_argument("--consider_task", default=DEFAULT_CONSIDER_TASK, help="Consider or not task in metric, default=" + str(DEFAULT_CONSIDER_TASK))
    parser.add_argument("--detach_logs", default=DEFAULT_DETACH_LOGS, help="Pass gradients or not through logarithms, default=" + str(DEFAULT_DETACH_LOGS))
    parser.add_argument("--n_parts", default=DEFAULT_N_PARTS, help="Number of parts in which the database is divided and store, default=" + str(DEFAULT_N_PARTS))
    parser.add_argument("--n_saves", default=DEFAULT_N_SAVES, help="Number of times the model is saved, default=" + str(DEFAULT_N_SAVES))
    parser.add_argument("--noisy", default=DEFAULT_NOISY, help="Use noisy layers in the concept module")
    parser.add_argument("--db_id", default=DEFAULT_DB_ID, help="Database ID")
    parser.add_argument("--id", default=DEFAULT_ID, help="ID of model to load")
    parser.add_argument("--inner_state_dim", default=DEFAULT_INNER_STATE_DIM, help="Dimensionality of inner state, default=" + str(DEFAULT_INNER_STATE_DIM))
    parser.add_argument("--n_tasks", default=DEFAULT_N_TASKS, help="Number of tasks, default=" + str(DEFAULT_N_TASKS))
    parser.add_argument("--n_concepts", default=DEFAULT_N_CONCEPTS, help="Number of concepts, default=" + str(DEFAULT_N_CONCEPTS))
    parser.add_argument("--n_actions", default=DEFAULT_N_ACTIONS, help="Number of actions, default=" + str(DEFAULT_N_ACTIONS))
    parser.add_argument("--n_batches", default=DEFAULT_N_BATCHES, type=int, help="Number of batches for estimation, default=" + str(DEFAULT_N_BATCHES))
    parser.add_argument("--lr", default=DEFAULT_LR, help="Learning rate, default=" + str(DEFAULT_LR))
    parser.add_argument("--update_rate", default=DEFAULT_UPDATE_RATE, help="Update rate for joint probability estimation, default=" + str(DEFAULT_UPDATE_RATE))
    parser.add_argument("--vision_latent_dim", default=DEFAULT_VISION_LATENT_DIM, help="Dimensionality of feature vector added to inner state, default=" + 
        str(DEFAULT_VISION_LATENT_DIM))
    args = parser.parse_args()

    project_name = 'visualSAC_conceptual_level'

    # Initilize Weights-and-Biases project
    wandb.init(project=project_name)

    # Log hyperparameters in WandB project
    wandb.config.update(args)

    device = 'cuda' if not args.cpu else 'cpu'
    
    database = load_database(args.n_parts, LOAD_PATH, args.db_id, args.buffer_size, 3)
    train_database, test_database = separate_database(database)

    conceptual_agent = Conceptual_Agent(args.inner_state_dim, args.vision_latent_dim, args.n_concepts, args.noisy, args.lr).to(device)
    if args.id is not None:
        conceptual_agent.load(SAVE_PATH, args.id)
    concept_optimizer = S_ConceptOptimizer(args.batch_size, args.beta, args.eta, args.n_batches, args.update_rate, args.consider_task, args.detach_logs)

    os.makedirs(SAVE_PATH, exist_ok=True)

    for step in range(0, args.n_steps):
        initialization = step==0
        train_metrics = concept_optimizer.optimize(conceptual_agent, train_database, args.n_actions, args.n_tasks, initialization, train=True)        
        test_metrics = concept_optimizer.optimize(conceptual_agent, test_database, args.n_actions, args.n_tasks, initialization, train=False)
        
        metrics = {**train_metrics, **test_metrics}
        metrics['step'] = step
        wandb.log(metrics)
    
        should_save = ((step+1) % (args.n_steps // args.n_saves)) == 0
        if should_save:
            v = (step+1) // (args.n_steps // args.n_saves)
            conceptual_agent.save(SAVE_PATH, v=v)
