# #######################################################
# ACKNOWLEDGEMENT 
# https://github.com/ikostrikov/pytorch-trpo
# #######################################################
import argparse
from itertools import count

import gym
import scipy.optimize
import os
import torch
import time
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from matplotlib import pyplot 
import matplotlib.pyplot as plt
import pickle

try:
    import safety_gym.envs
except ImportError:
    print("can not find safety gym...")

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--adv', action='store_true', help='adversary agent')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                    help='interval between training status logs (default: 10)')

args = parser.parse_args()
eval_epoch = 5
env = gym.make(args.env_name)
env.seed(args.seed)

test_env = gym.make(args.env_name)
test_env.seed(args.seed+10)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

torch.manual_seed(args.seed)
adv_agent = args.adv
if adv_agent:
    print("*"*20)
    print("Adversary agent")
    print("*"*20)

hidden_size = args.hidden_size
policy_net = Policy(num_inputs, num_actions, hidden_size)
value_net = Value(num_inputs, num_actions, hidden_size)

def plot_data(reward=None, cost=None, loss=None, path=None):
    if not reward==None:
        plt.figure()
        plt.plot(range(len(reward)), reward)
        plt.xlabel('No of iter:')
        plt.ylabel('Reward:')
        plt.savefig(path+'/reward_plot.png', format='png')
        plt.close()
    if not cost==None:
        plt.figure()
        plt.plot(range(len(cost)), cost)
        plt.xlabel('No of iter:')
        plt.ylabel('Cost:')
        plt.savefig(path+'/cost_plot.png', format='png')
        plt.close()
    if not loss==None:
        plt.figure()
        plt.plot(range(len(loss)), loss)
        plt.xlabel('No of iter:')
        plt.ylabel('Loss:')
        plt.savefig(path+'/loss_plot.png', format='png')
        plt.close()

def eval_episode(test_env, state_processor, max_episode_steps=400):
    num_steps = 0
    reward_sum = 0
    cost_sum = 0
    done = False
    state = test_env.reset()
    while not done:
        num_steps+=1
        state = state_processor(state)
        action = select_action(state)
        action = action.data[0].numpy()
        next_state, reward, done, info = test_env.step(action)
        reward_sum += reward
        if "cost" in info:
            cost = info["cost"]
            cost_sum += float(cost)
        state = next_state
        done = True if num_steps==max_episode_steps else done
        if done:
           break
    return reward_sum, cost_sum



def use_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states), Variable(actions))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states), Variable(actions))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    loss = get_loss()
    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)
    return loss

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)
######################################################
######################################################
######################################################
model_dir = os.getcwd() 
hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
if adv_agent:
    data_path = os.path.join(model_dir,'data', args.env_name, 'Adversary', hms_time)
else:
    data_path = os.path.join(model_dir,'data', args.env_name, 'Taskpolicy', hms_time)
config = dict()
config['env_name'] = args.env_name
config['obs_dim'] = num_inputs
config['action_dim'] = num_actions
config['hidden_size']= hidden_size
config_file = use_path(data_path+'/best_model')+'/config.pickle'
if not os.path.exists(config_file):
  with open(config_file, 'wb') as f:
      pickle.dump(config, f)

reward_vec = []
cost_vec = []
loss_vec = []

if adv_agent:
    best_reward = np.inf
    best_cost = -np.inf
else:
    best_reward = -np.inf
    best_cost = np.inf  
######################################################
######################################################
######################################################
for i_episode in count(1):
    memory = Memory()
    num_steps = 0
    reward_batch = 0
    cost_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        state = env.reset()
        state = running_state(state)
        reward_sum = 0
        cost_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            action = select_action(state)
            action = action.data[0].numpy()
            next_state, reward, done, info = env.step(action)
            reward_sum += reward
            if "cost" in info:
                cost = info["cost"]
                cost_sum += float(cost)
            next_state = running_state(next_state)
            mask = 1
            if done:
                mask = 0
            if adv_agent:
                memory.push(state, np.array([action]), mask, next_state, cost)
            else:
                memory.push(state, np.array([action]), mask, next_state, reward)

            if args.render:
                env.render()
            if done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum
        cost_batch += cost_sum

    reward_batch /= num_episodes
    cost_batch /=num_episodes
    batch = memory.sample()
    loss = update_params(batch)

    reward_vec.append(reward_batch)
    cost_vec.append(cost_batch)
    loss_vec.append(loss)
    plot_data(reward_vec, cost_vec, loss_vec, use_path(data_path+'/plot'))
    if i_episode % args.log_interval == 0:
        if adv_agent:
            print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(i_episode, cost_sum, cost_batch))
        else:
            print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(i_episode, reward_sum, reward_batch))
        reward_avg = 0
        cost_avg = 0
        for _ in range(eval_epoch):
           r, c = eval_episode(test_env, running_state)
           reward_avg+=r
           cost_avg+=c
        reward_avg = reward_avg/eval_epoch
        cost_avg = cost_avg/eval_epoch
        if adv_agent:
            print('Evaluation ->> Average Reward: {}\tAverage Cost {:.2f}'.format(reward_avg, cost_avg))
            if best_cost<=cost_avg:
                best_reward = reward_avg
                best_cost = cost_avg
                value_net.save(use_path(data_path+'/best_model_{}'.format(cost_avg)))
                policy_net.save(use_path(data_path+'/best_model_{}'.format(cost_avg)))
        else:
            print('Evaluation ->> Average Reward: {}\tAverage Cost {:.2f}'.format(reward_avg, cost_avg))
            if best_reward<=reward_avg:
                best_reward = reward_avg
                best_cost = cost_avg
                value_net.save(use_path(data_path+'/best_model_{}'.format(reward_avg)))
                policy_net.save(use_path(data_path+'/best_model_{}'.format(reward_avg)))


            value_net.save(use_path(data_path+'/best_model'))
