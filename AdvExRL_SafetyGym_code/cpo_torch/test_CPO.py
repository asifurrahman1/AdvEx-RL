#######################################################
# ACKNOWLEDGEMENT 
# https://github.com/dobro12/CPO/tree/master/torch
# #######################################################
from cpo_torch.logger import Logger
from cpo_torch.agent import Agent
from cpo_torch.graph import Graph
from cpo_torch.env import Env

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

class CPO_TEST:
    def __init__(self,
                 env_name,
                 mx_ep_len, 
                 seed=0,
                 cpo_model_dir=None,
                 adv_agent=None, 
                 adv_shield=None,
                 safety_policy=None, 
                 ):
          algo_idx = 1
          agent_name = 'CPO'
          algo = '{}_{}'.format(agent_name, algo_idx)
          model_dir = cpo_model_dir
          save_name = ''
          # save_name = model_dir+"/result/{}_{}".format(save_name, algo)

          args = {'agent_name':agent_name,
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
              self.device = torch.device('cuda:0')
              print('[torch] cuda is used.')
          else:
              self.device = torch.device('cpu')
              print('[torch] cpu is used.')
          self.max_ep_len = mx_ep_len
          self.seed = seed
          np.random.seed(self.seed)
          random.seed(self.seed)
          self.env = Env(env_name, self.seed, self.max_ep_len)
          self.agent = Agent(self.env, self.device, args)
          self.safety_policy = safety_policy
          self.adv_agent = adv_agent
          self.shield_agent = adv_shield

    def aaa_atk_eval_episode(self,
                     atk_rate=0,
                     aaa_atk=False,
                     use_safety=False,
                     safety_shield = False,
                     shield_threshold1=17.5,
                     shield_thershold2=None
                     ):
          num_steps = 0
          reward_sum = 0
          cost_sum = 0
          state = self.env.reset()
          done = False
          sfty_cnt=0
          tsk_cnt=0
          while not done:
              num_steps+=1
              #_________________________________________________
              #*************************************************
              state_tensor = torch.tensor(state, device=self.device, dtype=torch.float)
              action_tensor, clipped_action_tensor = self.agent.getAction(state_tensor, is_train=False)
              action = action_tensor.detach().cpu().numpy()
              action_tsk = clipped_action_tensor.detach().cpu().numpy()
              #*************************************************
              #_________________________________________________
              ###################################################
              #************************************************** 
              if np.random.rand()<atk_rate and aaa_atk:
                    action_tsk = self.adv_agent.select_action(self.env._env.obs())
              # #**************************************************   
              # ###################################################
              if use_safety:
                    if safety_shield:
                        shield_val_tsk = self.safety_policy.get_shield_value(self.env._env.obs(), action_tsk)
                    else: 
                        shield_val_tsk = self.adv_agent.get_shield_value(self.env._env.obs(), action_tsk)
                    # shield_val_tsk = self.safety_policy.get_shield_value(self.env._env.obs(), action_tsk)
                    # shield_val_tsk = adv_agent.get_shield_value(self.env._env.obs(), action_tsk)
                    if shield_thershold2==None:
                      if shield_val_tsk>=shield_threshold1:
                        action = self.safety_policy.select_action(self.env._env.obs(), eval=True)
                        sfty_cnt+=1
                      else:
                        action = action_tsk 
                        tsk_cnt+=1 
                    else:
                      if shield_val_tsk>=shield_threshold1 and shield_val_tsk<shield_thershold2:
                        action = self.safety_policy.select_action(self.env._env.obs(), eval=True)
                        sfty_cnt+=1
                      else:
                        action = action_tsk 
                        tsk_cnt+=1  
              else:
                  action = action_tsk 
                  tsk_cnt+=1      
              # #**************************************************   
              ###################################################
              next_state, reward, done, info = self.env.step(action)
              reward_sum += reward
              if "cost" in info:
                  cost = info["cost"]
                  cost_sum += float(cost)
              state = next_state
              done = True if num_steps==self.max_ep_len else done
              if done:
                epi_step_info=dict()
                epi_step_info['epi_step']=num_steps
                epi_step_info['task_count']=tsk_cnt
                epi_step_info['safety_count']=sfty_cnt
                break
          return reward_sum, cost_sum, epi_step_info


    def random_atk_eval_episode(self,
                     atk_rate=0,
                     random_atk=False,
                     use_safety=False,
                     safety_shield = False,
                     shield_threshold1=17.5,
                     shield_thershold2=None
                     ):
          num_steps = 0
          reward_sum = 0
          cost_sum = 0
          state = self.env.reset()
          done = False
          sfty_cnt=0
          tsk_cnt=0
          while not done:
              num_steps+=1
              #_________________________________________________
              #*************************************************
              state_tensor = torch.tensor(state, device=self.device, dtype=torch.float)
              action_tensor, clipped_action_tensor = self.agent.getAction(state_tensor, is_train=False)
              action = action_tensor.detach().cpu().numpy()
              action_tsk = clipped_action_tensor.detach().cpu().numpy()
              #*************************************************
              #_________________________________________________
              ###################################################
              #************************************************** 
              if np.random.rand()<atk_rate and random_atk:
                    action_tsk = self.env.action_space.sample()
              # #**************************************************   
              # ###################################################
              if use_safety:
                    if safety_shield:
                        shield_val_tsk = self.safety_policy.get_shield_value(self.env._env.obs(), action_tsk)
                    else: 
                        shield_val_tsk = self.adv_agent.get_shield_value(self.env._env.obs(), action_tsk)
                    if shield_thershold2==None:
                      if shield_val_tsk>=shield_threshold1:
                        action = self.safety_policy.select_action(self.env._env.obs(), eval=True)
                        # rec_cnt+=1
                      else:
                        action = action_tsk 
                        tsk_cnt+=1 
                    else:
                      if shield_val_tsk>=shield_threshold1 and shield_val_tsk<shield_thershold2:
                        action = self.safety_policy.select_action(self.env._env.obs(), eval=True)
                        # rec_cnt+=1
                      else:
                        action = action_tsk 
                        tsk_cnt+=1  
              else:
                  action = action_tsk 
                  tsk_cnt+=1      
              # #**************************************************   
              ###################################################
              next_state, reward, done, info = self.env.step(action)
              reward_sum += reward
              if "cost" in info:
                  cost = info["cost"]
                  cost_sum += float(cost)
              state = next_state
              done = True if num_steps==self.max_ep_len else done
              if done:
                epi_step_info=dict()
                epi_step_info['epi_step']=num_steps
                epi_step_info['task_count']=tsk_cnt
                epi_step_info['safety_count']=sfty_cnt
                break
          return reward_sum, cost_sum, epi_step_info


