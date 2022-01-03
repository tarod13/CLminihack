import os
import argparse

DEFAULT_ENV_NAME = 'AntGatherBoxes-v3'
DEFAULT_LEVEL = 3
DEFAULT_N_EPISODES = 600
DEFAULT_N_ITERS = 4
DEFAULT_REWARD_SCALE_MC = 0.1
DEFAULT_POLICY_DIVERGENCE_LIMIT = 0.05
DEFAULT_CONCEPT_MODEL_ID = '2021-02-19_20-52-34_v59' #'2021-02-19_20-52-36_v38' #'2021-02-19_20-52-34_v59'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default=DEFAULT_ENV_NAME, help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--level", default=DEFAULT_LEVEL, type=int, help="Training level, default=" + str(DEFAULT_LEVEL))
    parser.add_argument("--n_iters", default=DEFAULT_N_ITERS, type=int, help="Number of iterations, default=" + str(DEFAULT_N_ITERS))
    parser.add_argument("--n_episodes", default=DEFAULT_N_EPISODES, type=int, help="Number of episodes, default=" + str(DEFAULT_N_EPISODES))
    parser.add_argument("--policy_divergence_limit", type=float, default=DEFAULT_POLICY_DIVERGENCE_LIMIT, help="Max divergence between old and new MC policy, default=" + str(DEFAULT_POLICY_DIVERGENCE_LIMIT))
    parser.add_argument("--reward_scale_MC", default=DEFAULT_REWARD_SCALE_MC, help="MC reward scale, default=" + str(DEFAULT_REWARD_SCALE_MC))
    parser.add_argument("--load_concept_id", default=DEFAULT_CONCEPT_MODEL_ID, help="ID of concept model to load")
    args = parser.parse_args()

    for iter in range(0, args.n_iters):
        if args.level == 2:
            os.system("python training_2l.py --env_name "+args.env_name+" --n_episodes {}".format(args.n_episodes))
        elif args.level == 3:
            os.system("python training_3l.py --env_name "+args.env_name+" --n_episodes {}".format(args.n_episodes)+
                        " --reward_scale_MC {}".format(args.reward_scale_MC)+" --load_concept_id "+args.load_concept_id+
                        " --policy_divergence_limit {}".format(args.policy_divergence_limit))
        else:
            raise RuntimeError("Invalid level")