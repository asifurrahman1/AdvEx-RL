# #######################################################
# ACKNOWLEDGEMENT 
# https://github.com/ikostrikov/pytorch-trpo
# #######################################################
import torch
import torch.autograd as autograd
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden=256):
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

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def save(self, path):
         model_path = path+'/policy.pth'
         torch.save(self.state_dict(), model_path)

    def load(self, path, device=None):
        self.load_state_dict(torch.load(path, map_location=device))

class Value(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(obs_dim+action_dim, hidden)
        self.affine2 = nn.Linear(hidden, hidden)
        self.value_head = nn.Linear(hidden, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, obs, act):
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
