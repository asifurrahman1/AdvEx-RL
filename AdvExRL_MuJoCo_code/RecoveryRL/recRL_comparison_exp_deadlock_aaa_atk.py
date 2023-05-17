import pickle
import torch
from RecoveryRL.recoverRL_args import recRL_get_args
import os
import sys
import numpy as np
import datetime
from RecoveryRL.recoveryRL_agent import Recovery_SAC
import copy
from AdvEx_RL.sac import SAC
from tqdm import tqdm


# TORCH_DEVICE = torch.device(
#     'cuda') if torch.cuda.is_available() else torch.device('cpu')


def torchify(x, device=None):
    if not device==None:
      return torch.FloatTensor(x).to(device)
    else:
       device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
       return torch.FloatTensor(x).to(device)

class Comparative_Experiment():
        def __init__(self, env, exp_cfg, expert_path, recovery_path, adv_path, adv_cfg=None, device=None, step_look_ahead=10):
            self.exp_cfg = exp_cfg
            if adv_cfg==None:
                self.adv_cfg = exp_cfg
            else:
                self.adv_cfg = adv_cfg
            #***********************************
            self.env = env
            self.agent_observation_space = env.observation_space.shape[0]
            self.agent_action_space = env.action_space.shape[0]
            self.step_look_ahead = step_look_ahead
            #***********************************
            #***********************************
            self.expert_path = expert_path
            self.adv_path = adv_path
            self.recovery_path = recovery_path
            if not device ==None:
              self.device = device
            else:
              self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            #***********************************
            #***********************************
            self.agent = Recovery_SAC(self.agent_observation_space,
                                      self.agent_action_space ,
                                      self.exp_cfg,
                                      env=self.env,
                                      device=self.device
                                     )
            self.agent.load_models(self.expert_path, self.recovery_path, self.device)
            #***********************************
            logdir = ''
            self.aaa_agent = SAC(self.agent_observation_space,
                                    self.agent_action_space,
                                    self.adv_cfg,
                                    logdir,
                                    env=self.env,
                                    device = self.device
                                    )
            self.aaa_agent.load_best_model(self.adv_path)
            

        def get_action(self, env, state, atk_rate, epsilon,  aaa= True):
                def recovery_thresh(state, action):
                    if not self.exp_cfg.use_recovery:
                        return False
                    critic_val = self.agent.safety_critic.get_value(
                        torchify(state).unsqueeze(0),
                        torchify(action).unsqueeze(0))
                    if critic_val > self.exp_cfg.eps_safe:
                        return True
                    return False
                action = self.agent.select_action(state, eval=True)  # Sample action from policy
                #******************************************************************************
                if np.random.rand()<atk_rate and aaa:
                    action = self.aaa_agent.select_action(state, eval=True)
                #******************************************************************************           
                if recovery_thresh(state, action) and self.exp_cfg.use_recovery:
                    recovery = True
                    real_action = self.agent.safety_critic.select_action(state)
                else:
                    recovery = False
                    real_action = np.copy(action)
                return action, real_action, recovery

        def run_test_experiment(self, n_episode, atk_rate, epsilon, ):
            
            rectsk_safety_vec = []
            rectsk_tsk_cnt_vec = []
            rectsk_rec_cnt_vec = []
            rectsk_reward_vec = []
            rectsk_cycle_vec = []
            rectsk_info_vec = []
            for i in tqdm(range(n_episode)):
                env1 = copy.deepcopy(self.env)
                env2 = copy.deepcopy(self.env)
                
                rec_epi_rec_safety, rec_epi_tsk_cnt, rec_epi_rec_cnt, rec_epi_reward, cycle_count, rectsk_info = self.test_roll_out(env2, atk_rate, epsilon, aaa_atk= True, deadlock_look_ahead=self.step_look_ahead)
                rectsk_safety_vec.append(rec_epi_rec_safety)
                rectsk_tsk_cnt_vec.append(rec_epi_tsk_cnt)
                rectsk_rec_cnt_vec.append(rec_epi_rec_cnt)
                rectsk_reward_vec. append(rec_epi_reward)
                rectsk_cycle_vec.append(cycle_count)
                rectsk_info_vec.append(rectsk_info)
            
            # task_agent = { 'reward': taskonly_reward_vec,
            #                'safety': taskonly_safety_vec,
            #                'adv_reward': taskonly_adv_reward_vec,
            #                'tsk_cnt': taskonly_tsk_cnt_vec,
            #                'info': taskonly_info_vec
            #             }
            task_rec_agent = {'reward': rectsk_reward_vec,
                              'safety': rectsk_safety_vec,
                              'cycle_count': rectsk_cycle_vec,
                              'tsk_cnt': rectsk_tsk_cnt_vec,
                              'rec_cnt': rectsk_rec_cnt_vec,
                              'info':rectsk_info_vec
                             }

            Data = { 'task_rec_agent': task_rec_agent
                    }
            return Data

        def test_roll_out(self, 
                          env, 
                          atk_rate=0.00, 
                          epsilon=0.00,
                          task_only = False,
                          aaa_atk = False,
                          deadlock_look_ahead=10
                          ):
            done = False
            reward = 0
            epi_step = 0
            epi_reward = 0
            adv_reward = 0
            safety = 0
            ######### BUFFER FOR DEADLOCK DETECTION########
            deadlock_detection_vec = []
            buffer_cap = deadlock_look_ahead
            buff_index = 0
            no_of_cycle = 0
            ###############################################

            rec_cnt = 0
            tsk_cnt = 0
            info_vec = []
            state = self.env.reset()
            while not done:
                epi_step+=1
                ##############################################
                if len(deadlock_detection_vec) < buffer_cap:
                      deadlock_detection_vec.append(None)
                deadlock_detection_vec[buff_index]=(list(state))
                buff_index = (buff_index+1)%buffer_cap
                ##############################################
                action, final_action, recovery_selected = self.get_action(env, state, atk_rate=atk_rate, epsilon= epsilon, aaa=aaa_atk)
                if (not task_only) and recovery_selected:
                    rec_cnt+=1
                else:
                    tsk_cnt+=1

                if task_only:
                    next_state, reward, done, info = self.env.step(action)  # Step
                else:
                    next_state, reward, done, info = self.env.step(final_action)  # Step
                info_vec.append(info)
                ######### DEAD LOCK DETECTION ##################
                if recovery_selected:
                      if list(next_state) in deadlock_detection_vec:
                            no_of_cycle +=1
                ################################################
                epi_reward+=reward
                adv_reward+=info['adv_reward']
                done = done or epi_step == env._max_episode_steps or (float(info['adv_reward'])>0)
                state = next_state
                if done:
                    if adv_reward<0:
                        safety=1
                    else:
                        safety = epi_step/env._max_episode_steps
                    break
            return safety, tsk_cnt, rec_cnt, epi_reward, no_of_cycle, info_vec



def set_algo_configuration(algo, args):
    if algo=='RRL_MF':
        args.cuda = True
        args.seed = 1234
        args.use_recovery=True
        args.MF_recovery= True
        args.Q_sampling_recovery = False
        args.gamma_safe = 0.8
        args.eps_safe = 0.3

    elif algo=='unconstrained':
        args.use_recovery=False
        args.MF_recovery= False
        args.cuda = True
        args.seed = 1234
        
    elif algo=='RSPO':
        args.cuda = True
        args.seed = 1234
        args.use_recovery=False
        args.MF_recovery= False
        args.DGD_constraints= True
        args.gamma_safe = 0.8
        args.eps_safe = 0.3

    elif algo=='RCPO':
        args.cuda = True
        args.use_recovery=False
        args.MF_recovery= False
        args.seed = 1234
        args.gamma_safe = 0.8
        args.eps_safe = 0.3
        args.RCPO = True

    elif algo=='RP':
        args.cuda = True
        args.use_recovery=False
        args.MF_recovery= False
        args.seed = 1234
        args.constraint_reward_penalty = 1000

    elif algo=='SQRL':
        args.cuda = True
        args.seed = 1234
        args.use_constraint_sampling= True
        args.use_recovery=False
        args.MF_recovery= False
        args.DGD_constraints= True
        args.gamma_safe = 0.8
        args.eps_safe = 0.3

    elif algo=='LR':
        args.cuda = True
        args.use_recovery=False
        args.MF_recovery= False
        args.seed = 1234
        args.DGD_constraints= True
        args.gamma_safe = 0.8
        args.eps_safe = 0.3
    return args

def get_model_directories(logdir):
    algo = []
    experiment_map = {}
    experiment_map['algos'] = {}
    best_agent_path = None
    
    for fname in os.listdir(logdir):
      best_agent = -999999999
      algo.append(fname)
      agent_path = os.path.join(logdir, fname, 'sac_agent', 'best') 
    #   print(agent_path)
      for model in os.listdir(agent_path):
          name = model.split('reward_')[-1]
          if best_agent<=float(name):
            best_agent = float(name)
            best_agent_path = os.path.join(agent_path, model) 
      
      recovery_path = os.path.join(logdir, fname, 'safety_critic_recovery')
      for rec_model in os.listdir(recovery_path):
          recovery_agent_path = os.path.join(recovery_path, rec_model)
      result = None
      agents_path = {'agent': best_agent_path,
                    'recovery_agent': recovery_agent_path ,
                    'result': result
                    } 
      experiment_map['algos'][fname] = agents_path
      
    return experiment_map 
    
def run_comparison(env, env_model_path, atk_rt=0.0, eps=0.0, aaa_agent_path=None, aaa_cfg=None, eval_episode = 100, device=None, look_ahead=10):
    args = recRL_get_args()
    env = env
    atk_rate = atk_rt
    epsilon = eps
    eval_episode = eval_episode
    algo_model_path_map = get_model_directories(env_model_path)
    for algo in algo_model_path_map['algos']:
       args = set_algo_configuration(algo, args)
       expert_path =  algo_model_path_map['algos'][algo]['agent']
       recovery_path = algo_model_path_map['algos'][algo]['recovery_agent']
       evaluate_algo = Comparative_Experiment(env, args, expert_path, recovery_path, aaa_agent_path, aaa_cfg, device=device, step_look_ahead=look_ahead)
       exp_data = evaluate_algo.run_test_experiment(eval_episode, atk_rate=atk_rate, epsilon= epsilon)
       
       algo_model_path_map['algos'][algo]['result'] = exp_data
    return algo_model_path_map
