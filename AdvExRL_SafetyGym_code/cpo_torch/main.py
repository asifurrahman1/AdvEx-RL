#######################################################
# ACKNOWLEDGEMENT 
# https://github.com/dobro12/CPO/tree/master/torch
# #######################################################
from logger import Logger
from agent import Agent
from graph import Graph
from env import Env

from sklearn.utils import shuffle
from collections import deque
from scipy.stats import norm
from copy import deepcopy
import numpy as np
import safety_gym
import argparse
import pickle
import random
import torch
import wandb
import copy
import time
import gym
from tqdm import tqdm

try:
    import safety_gym.envs
except ImportError:
    print("can not find safety gym...")

def train(main_args):
    algo_idx = 1
    agent_name = 'CPO'
    env_name = main_args.env_name
    max_ep_len = 1000
    max_steps = 4000
    epochs = 2500
    save_freq = 10
    algo = '{}_{}'.format(agent_name, algo_idx)
    save_name = '_'.join(env_name.split('-')[:-1])
    save_name = "result/{}_{}".format(save_name, algo)
    args = {
        'agent_name':agent_name,
        'save_name': save_name,
        'discount_factor':0.99,
        'hidden1':512,
        'hidden2':512,
        'v_lr':2e-4,
        'cost_v_lr':2e-4,
        'value_epochs':200,
        'batch_size':10000,
        'num_conjugate':10,
        'max_decay_num':10,
        'line_decay':0.8,
        'max_kl':0.001,
        'damping_coeff':0.01,
        'gae_coeff':0.97,
        'cost_d':25.0/1000.0,
    }
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')

    # for random seed
    seed = algo_idx + random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)

    env = Env(env_name, seed, max_ep_len)
    agent = Agent(env, device, args)

    # for wandb
    wandb.init(project='[torch] CPO')
    if main_args.graph: graph = Graph(1000, "TRPO", ['score', 'cv', 'policy objective', 'value loss', 'kl divergence', 'entropy'])

    for epoch in range(epochs):
        trajectories = []
        ep_step = 0
        scores = []
        cvs = []
        while ep_step < max_steps:
            state = env.reset()
            score = 0
            cv = 0
            step = 0
            while True:
                ep_step += 1
                step += 1
                state_tensor = torch.tensor(state, device=device, dtype=torch.float)
                action_tensor, clipped_action_tensor = agent.getAction(state_tensor, is_train=True)
                action = action_tensor.detach().cpu().numpy()
                clipped_action = clipped_action_tensor.detach().cpu().numpy()
                next_state, reward, done, info = env.step(clipped_action)
                cost = info['cost']

                done = True if step >= max_ep_len else done
                fail = True if step < max_ep_len and done else False
                trajectories.append([state, action, reward, cost, done, fail, next_state])

                state = next_state
                score += reward
                cv += info['num_cv']

                if done or step >= max_ep_len:
                    break

            scores.append(score)
            cvs.append(cv)

        v_loss, cost_v_loss, objective, cost_surrogate, kl, entropy = agent.train(trajs=trajectories)
        score = np.mean(scores)
        cvs = np.mean(cvs)
        log_data = {"score":score, 'cv':cv, "value loss":v_loss, "cost value loss":cost_v_loss, "objective":objective, "cost surrogate":cost_surrogate, "kl":kl, "entropy":entropy}
        print(log_data)
        if main_args.graph: graph.update([score, objective, v_loss, kl, entropy])
        wandb.log(log_data)
        if (epoch + 1)%save_freq == 0:
            agent.save()

    if main_args.graph: graph.update(None, finished=True)


def test_CPO(env_name, 
             mx_ep_len, 
             epoch, 
             cpo_model_dir=None,
             adv_agent=None, 
             adv_shield=None,
             safety_policy=None, 
             aaa_atk=False,
             use_safety=False,
             ):
    algo_idx = 1
    agent_name = 'CPO'
    max_ep_len = mx_ep_len
    epochs = epoch
    algo = '{}_{}'.format(agent_name, algo_idx)
    save_name = ''
    args = {
        'agent_name':agent_name,
        'save_name': save_name,
        'discount_factor':0.99,
        'hidden1':512,
        'hidden2':512,
        'v_lr':2e-4,
        'cost_v_lr':2e-4,
        'value_epochs':200,
        'batch_size':10000,
        'num_conjugate':10,
        'max_decay_num':10,
        'line_decay':0.8,
        'max_kl':0.001,
        'damping_coeff':0.01,
        'gae_coeff':0.97,
        'cost_d':25.0/1000.0,
    }
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')

    # for random seed
    seed = algo_idx + random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)
    env = Env(env_name, seed, max_ep_len)
    agent = Agent(env, device, args)
    ###########################################
    agent.load()
    ###########################################
    score_vec = []
    costs_vec = []
    for epoch in tqdm(range(epochs)):
        state = env.reset()
        done = False
        scores = 0
        costs = 0
        step = 0
        ep_step = 0
        while not done:
            ep_step += 1
            step += 1
            state_tensor = torch.tensor(state, device=device, dtype=torch.float)
            action_tensor, clipped_action_tensor = agent.getAction(state_tensor, is_train=True)
            action = action_tensor.detach().cpu().numpy()
            clipped_action = clipped_action_tensor.detach().cpu().numpy()
            next_state, reward, done, info = env.step(clipped_action)
            cost = info['cost']
            costs+=cost
            scores += reward
            done = True if step >= max_ep_len else done
            fail = True if step < max_ep_len and done else False
            state = next_state
            if done or step >= max_ep_len:
                score_vec.append(scores)
                costs_vec.append(costs)
                break
    return score_vec, costs_vec




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPO')
    parser.add_argument('--env-name', default='none', help='Set test environment to setup all configuration')
    parser.add_argument('--test', action='store_true', help='For test.')
    parser.add_argument('--resume', type=int, default=0, help='type # of checkpoint.')
    parser.add_argument('--graph', action='store_true', help='For graph.')
    args = parser.parse_args()
    dict_args = vars(args)
    if args.test:
        pass
    else:
        train(args)
