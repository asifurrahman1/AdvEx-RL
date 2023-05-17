import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os.path as osp
import math
import numpy as np
from torch.distributions import Normal
import torch
import torch.nn.functional as F
from datetime import datetime
from torch.autograd import Variable
from torch.optim import Adam, SGD
from AdvExRL_safetygym.network import Qvalue_Network, Double_QNetwork, StochasticPolicy, grad_false
import sys
import pickle

class Safety_Agent(object):
        def __init__(self,
                    observation_space,
                    action_space,
                    args,
                    logdir,
                    env= None,
                    adv_agent_path = None,
                    im_shape = None,
                    temp_env = None,
                    policy_model_path=None,
                    adv_agent = None,
                    device = None
                    ):
            self.learning_steps = 0
            self.action_space = action_space
            self.observation_space = observation_space
            self.env = env
            self.cfg = args
            #---------------------------------------------
            #self.policy_type = args.policy    #Gaussian   Deterministic #else Stochastic
            self.gamma = args.gamma
            self.tau = args.tau
            self.alpha = args.alpha
            self.env_name = args.env_name
            self.adv_agent_path = adv_agent_path
            
            self.target_update_interval = args.target_update_interval
            if device==None:
                self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            else:
                self.device = device

            self.logdir = logdir
            self.adv_critic = None
            #=======================================================================
            if not adv_agent ==None:
                self.adv_critic = adv_agent
            #---------------------------------------------------------------------
            # self.safety_critic = Qvalue_Network(observation_space,
            #                                     action_space,
            #                                     self.cfg.hidden_size
            #                                     ).to(device=self.device)
            self.safety_critic = Double_QNetwork(observation_space,
                                                action_space,
                                                self.cfg.hidden_size
                                                ).to(device=self.device)

            self.safety_policy = StochasticPolicy(self.observation_space,
                                                  self.action_space,
                                                  self.cfg.hidden_size,
                                                  action_space=self.env.action_space
                                                 ).to(self.device)

            self.safety_critic_optim = Adam(self.safety_critic.parameters(), lr=args.lr)
            self.safety_policy_optim = Adam(self.safety_policy.parameters(), lr=args.lr)
        #=====================================================================
        #=====================================================================
        def torchify(self, x): 
         return torch.FloatTensor(x).to(self.device).unsqueeze(0)

        # def get_shield_value(self, state, action):
        #     state = torch.from_numpy(state).type(torch.FloatTensor)
        #     action = torch.from_numpy(action).type(torch.FloatTensor)
        #     with torch.no_grad():
        #         q = self.safety_critic(Variable(state), Variable(action))
        #     return q.cpu().detach().numpy()[0]

        def get_shield_value(self, state, action):
            with torch.no_grad():
                q1, q2 = self.safety_critic(self.torchify(state), self.torchify(action))
            return torch.max(q1, q2)
            
        #************************************************************************************
        #****************ABLATION TEST WITH SHIELD+RANDOM POLICY************************************
        def select_ablation_action(self):
            action = self.env.action_space.sample()
            return action
        #************************************************************************************
        #************************************************************************************

        #************************************************************************************
        #*****************AdvEx-RL Safety Policy*******************************
        def select_action(self, state, eval=False):
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            if eval is False:
                action, _, _,_ = self.safety_policy.sample(state)
            else:
                _, _, action,_ = self.safety_policy.sample(state)
            action = action.detach().float().cpu().numpy()
            # action = np.clip(action[0], self.env.action_space.low.min(), self.env.action_space.high.max())
            return action[0]
        #************************************************************************************
        #************************************************************************************
        
        def get_batch_tensor(self, memory, batch_size, constrained=False):
            if not constrained:
                state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
                state_batch = torch.FloatTensor(state_batch).to(self.device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)
                reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
                mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
                return state_batch, action_batch, next_state_batch, reward_batch, mask_batch
            else:
                state_batch, action_batch, reward_batch, contraint_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
                state_batch = torch.FloatTensor(state_batch).to(self.device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)
                reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
                contraint_batch = torch.FloatTensor(contraint_batch).to(self.device).unsqueeze(1)
                mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
                return state_batch, action_batch, next_state_batch, reward_batch, contraint_batch, mask_batch
            
        #*********************UPDATE AdvEx-RL POLICY ******************************
        def update_safety_policy(self, memory, batch_size):
            state_batch, action_batch, next_state_batch, reward_batch, contraint_batch, mask_batch = self.get_batch_tensor(memory, batch_size, constrained=True)
            
            sampled_action, _, _, _ = self.safety_policy.sample(next_state_batch)
            next_state_q_risk = self.safety_critic(next_state_batch,sampled_action)
            
            with torch.no_grad():
                qf_pi = self.adv_critic.critic(next_state_batch, sampled_action)
            target_risk = contraint_batch+qf_pi
            est_err = F.mse_loss(target_risk, next_state_q_risk) 
            #==========Critic Loss Optim=======================
            self.safety_critic_optim.zero_grad()
            (est_err).backward()
            self.safety_critic_optim.step()
            #---------------------------------------------------
            curr_sampled_action, log_pi_agent, mean_rec, std_rec = self.safety_policy.sample(state_batch)
            q_risk = self.safety_critic(state_batch,curr_sampled_action)
            with torch.no_grad():
                _,log_pi_adv, mean_adv, std_adv = self.adv_critic.policy.sample(state_batch)
           
            log_pi_adv = log_pi_adv.float()
            log_pi_agent = log_pi_agent.float().unsqueeze(1)   
    
            target_max_sqf_pi = -(log_pi_agent-(q_risk-log_pi_adv))  
            safety_policy_loss = target_max_sqf_pi.mean() 
            
            self.safety_policy_optim.zero_grad()
            (safety_policy_loss).backward()
            self.safety_policy_optim.step()
            return safety_policy_loss.item()
        #************************************************************************************
        #************************************************************************************
       
        def update_parameters(self, memory, batch_size, nu=None, safety_critic=None):
            self.learning_steps+=1
            safety_policy_loss = self.update_safety_policy(memory, batch_size)
            return  safety_policy_loss

    
        def save_best_safety_model(self, ratio=0, interval=0):
            time = datetime.now().strftime("%b-%d-%Y")
            if not interval==0:
                model_dir = os.path.join(self.logdir,'AdvExRL_model/Interval','AdvExRL_at_interval_{}_safety_{}'.format(interval, ratio))
            else:
                model_dir = os.path.join(self.logdir,'AdvExRL_model/Best','{}_Best_safety_ratio{}'.format( time, ratio))
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            policy_path = os.path.join(model_dir, 'advexrl_policy_net.pth')
            self.safety_policy.save(policy_path)

            critic_path = os.path.join(model_dir, 'advexrl_critic_net.pth')
            self.safety_critic.save(critic_path)  
    

        def load_safety_model(self, path):
                safety_policy_path = path
                safety_policy_path_policy = os.path.join(safety_policy_path, 'advexrl_policy_net.pth')
                safety_critic_path_critic = os.path.join(safety_policy_path, 'advexrl_critic_net.pth')
                self.safety_policy.load(safety_policy_path_policy,  self.device)
                self.safety_critic.load(safety_critic_path_critic,  self.device)
                grad_false(self.safety_policy)
                grad_false(self.safety_critic)

            


          






