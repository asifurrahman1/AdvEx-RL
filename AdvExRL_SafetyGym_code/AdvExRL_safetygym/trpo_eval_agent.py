import numpy as np
import torch
from collections import deque
import torch.nn.functional as F
import pickle
from torch.autograd import Variable
from AdvExRL_safetygym.network import Qvalue_Network, Policy, grad_false
from pytorch_trpo.running_state import ZFilter

class TRPO(object):
    def __init__(self,
                 observation_shape=None,
                 action_shape=None,
                 model_dir=None,
                 action_space=None,
                 device=None
                 ):
            # self.running_state = ZFilter((observation_shape,), clip=5)
            if not device==None:
                self.device = device
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.running_state = ZFilter((observation_shape), clip=5)
            if not model_dir==None:
                  self.model_path = model_dir
                  critic_path = self.model_path+'/critic.pth'
                  policy_path = self.model_path+'/policy.pth'
                  with open(self.model_path+'/config.pickle', 'rb') as f:
                      config = pickle.load(f)
                  self.policy = Policy(config['obs_dim'],
                                      config['action_dim'],
                                      config['hidden_size'],
                                      action_space=action_space
                                      )
                  self.policy.load(policy_path, self.device)
                  self.critic = Qvalue_Network(config['obs_dim'],
                                              config['action_dim'],
                                              config['hidden_size'],
                                              )
                  self.critic.load(critic_path, self.device)
                  grad_false(self.policy)
                  grad_false(self.critic)    

    def get_shield_value(self, obs, action):
        obs = torch.from_numpy(obs).type(torch.FloatTensor)
        action = torch.from_numpy(action).type(torch.FloatTensor)
        with torch.no_grad():
            q = self.critic(Variable(obs), Variable(action))
        return q.cpu().detach().numpy()[0]

    def select_action(self, state):
            state = self.running_state(state)
            state = torch.from_numpy(state).unsqueeze(0).type(torch.FloatTensor)
            action_mean, _, action_std = self.policy(Variable(state))
            action = torch.normal(action_mean, action_std)
            action = action.data[0].numpy()    
            return action
     




          


