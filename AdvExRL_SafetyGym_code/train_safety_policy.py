import gym
import scipy.optimize
import os
import torch
import time
from AdvExRL_safetygym.safety_config import get_safety_args
from AdvExRL_safetygym.safety_trainer import Safety_trainer
import torch
import os
import argparse

try:
    import safety_gym.envs
except ImportError:
    print("can not find safety gym...")


def run(args):
    print(args.configure_env)
    env = gym.make(args.configure_env)
    env.seed(args.seed)

    env_sfty = gym.make(args.configure_env)
    env_sfty.seed(args.seed)

    test_env = gym.make(args.configure_env)
    test_env.seed(args.seed+10)

    safety_cfg = get_safety_args() 
    current_path = os.getcwd()
    args.logdir = current_path+'/data/{}/AdvEx_RL_Trained_Safety_policy/'.format(args.configure_env)
    exp = Safety_trainer(env, env_sfty, safety_cfg, test_env)
    exp.agent_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configure-env', default='none', help='Set test environment to setup all configuration')
    parser.add_argument('--seed', type=int, default=123, help='Set test environment to setup all configuration')
    args = parser.parse_args()
    run(args)
