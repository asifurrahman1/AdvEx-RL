import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# Soft update of target critic network
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden=256, action_space=None):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden)
        self.affine2 = nn.Linear(hidden, hidden)

        self.action_mean = nn.Linear(hidden, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, xin):
        # print(xin.shape)
        xin = xin
        # print(xin.dtype)
        x = torch.tanh(self.affine1(xin))
        # print(x.dtype)
        x = torch.tanh(self.affine2(x))
        # print(x.dtype)

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


    def sample(self, state):
        act_mean, act_log_std, act_std = self.forward(state)
        normal = Normal(act_mean, act_std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        mean = torch.tanh(act_mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, act_log_std


    def save(self, path):
         model_path = path+'/policy.pth'
         torch.save(self.state_dict(), model_path)

    def load(self, path, device=None):
        self.load_state_dict(torch.load(path, map_location=device))

class Double_QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256):
        super(Double_QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear5 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = nn.Linear(hidden_dim, 1)
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1).float()
        # print(xu)
        # print(xu.dtype)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear3(x1))
        x1 = self.linear4(x1)

        x2 = F.relu(self.linear5(xu))
        x2 = F.relu(self.linear6(x2))
        x2 = F.relu(self.linear7(x2))
        x2 = self.linear8(x2)

        return x1, x2

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.device = device
        self.load_state_dict(torch.load(path, map_location=device))

class Qvalue_Network(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super(Qvalue_Network, self).__init__()
        self.affine1 = nn.Linear(obs_dim+action_dim, hidden)
        self.affine2 = nn.Linear(hidden, hidden)
        self.value_head = nn.Linear(hidden, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, obs, act):
        # print(obs.shape)
        # print(act.shape)
        # print(obs.dtype)
        # print(act.dtype)
        obs = obs.type(torch.FloatTensor)
        act = act.type(torch.FloatTensor)
        # print(obs.dtype)
        # print(act.dtype)
        xu = torch.cat([obs, act], dim=-1)
        x = torch.tanh(self.affine1(xu))
        x = torch.tanh(self.affine2(x))
        state_values = self.value_head(x)
        return state_values

    def save(self, path):
        model_path = path+'/critic.pth'
        torch.save(self.state_dict(), model_path)

    def load(self, path, device=None):
        self.load_state_dict(torch.load(path, map_location=device))


class StochasticPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256, action_space=None):
        super(StochasticPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = torch.nn.Parameter(
            torch.as_tensor([np.log(0.1)] * num_actions))
        self.min_log_std = np.log(1e-6)
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.apply(weights_init_)
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        #print(self.log_std)
        log_std = torch.clamp(self.log_std, min=self.min_log_std)
        log_std = log_std.unsqueeze(0).repeat([len(mean), 1])
        std = torch.exp(log_std)
        return Normal(mean, std)

    def sample(self, state):
        dist  = self.forward(state)
        action = dist.rsample()
        return action, dist.log_prob(action).sum(-1), dist.mean, dist.stddev

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(StochasticPolicy, self).to(device)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.device = device
        self.load_state_dict(torch.load(path, map_location=device))