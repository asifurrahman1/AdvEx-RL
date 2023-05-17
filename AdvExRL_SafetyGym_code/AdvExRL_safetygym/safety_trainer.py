from datetime import datetime
import gym
import os
import os.path as osp
import pickle
import numpy as np
import itertools
import torch
import cv2
from torch import nn, optim
from trpo_eval_agent import TRPO
from safety_policy import Safety_Agent
from buffer_memory import  ConstraintReplayMemory
import copy
from matplotlib import pyplot 
import matplotlib.pyplot as plt
import os
import gym

try:
    import safety_gym.envs
except ImportError:
    print("can not find safety gym...")

# TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# def torchify(x , device): 
#   return torch.FloatTensor(x).to(TORCH_DEVICE)

class Safety_trainer():
#******************************************************************************************************************
    def __init__(self, env, env_safety, cfg, test_env):
        self.cfg = cfg
      #**************************************************************
        self.agent_observation_space = env.observation_space.shape[0]
        self.agent_action_space = env.action_space.shape[0]
        print(self.agent_observation_space)
        print(self.agent_action_space)
        
        current_path = os.getcwd()
        self.expert_agent_path = cfg.adv_path
        self.adv_agent_path = cfg.task_path
      #**************************************************************
        self.env = env
        self.env_sfty = env_safety
        self.test_env = test_env
       
        time = datetime.now().strftime("%b-%d-%Y|%H:%M|%p")
        self.logdir = os.path.join(self.cfg.logdir, self.cfg.configure_env, 'Model_trained_at_{}'.format(time))

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        
        print("Created LOGDIR: ", self.logdir)
        pickle.dump(self.cfg, open(os.path.join(self.logdir, "safety_args.pkl"), "wb"))
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        #-------------------------------------------------------------------
        # if self.cfg.env_name == "nav2": 
        ###################################################################
        ################## TRPO POLICY ####################################
        self.expert_agent = TRPO(model_dir = self.expert_agent_path,
                                 device = self.device,
                                 action_space=self.env.action_space
                                 )
                                 
        self.adv_agent = TRPO(model_dir = self.adv_agent_path,
                              device = self.device,
                              action_space=self.env.action_space
                              )                     
        ###################################################################
        #-------------------------------------------------------------------
        self.safety_agent = Safety_Agent(observation_space = self.agent_observation_space, 
                                         action_space= self.agent_action_space,
                                         args=self.cfg,
                                         logdir=self.logdir,
                                         env = self.env,
                                         adv_agent=self.adv_agent,
                                         device = self.device
                                         #********************
                                         )
        #------------------------------------------------------------------
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.env.seed(self.cfg.seed)
        self.test_env.seed(self.cfg.seed)
        #++++++++++++++++++++++++++++++++++++++++++
        self.safety_constrained_memory = ConstraintReplayMemory(10000, self.cfg.seed)
        #++++++++++++++++++++++++++++++++++++++++++
        self.no_of_training_episode = self.cfg.num_eps
        self.total_numsteps = 0
        self.num_viols = 0
        self.num_successes = 0

        self.best_cost = np.inf
        self.best_reward = -np.inf

        self.best_step_count = 0
        self.episode_count = 0
        self.safety_training_epoch = 10
        
        self.shield_theshold = self.cfg.shield_threshold #0.93 #.98
        self.beta = self.cfg.beta
        self.eta =  self.cfg.eta
#******************************************************************************************************************
        
    def agent_training(self):
        safety_pol_loss_vec = []
        safety_policy_cost = []
        safety_policy_reward = []
        task_count = []
        safety_count = []
        task_safety_reward = []
        task_safety_cost = []
        only_tsk_reward_vec = []
        only_tsk_cost_vec = []
        Flag_train_start = False
        warmup = False
        # print(self.no_of_training_episode)
        for n in range(self.no_of_training_episode):
            safety_reward, safety_cost, safety_step = self.run_safety_train_episode(warmup)
            print(f'Episode: {n}  '
                  f'Safety reward: {safety_reward:<6.2f}  '
                  f'Safety cost: {safety_cost:<6.2f}  '
                  f'Safety step: {safety_step:<6.2f}  '
                  )
            if (len(self.safety_constrained_memory) > self.safety_training_epoch*self.cfg.batch_size) and (n>=self.cfg.train_start):
                if (len(self.safety_constrained_memory) > self.safety_training_epoch*self.cfg.batch_size) and not Flag_train_start:
                    print("Training Started")   
                    warmup=False
                    Flag_train_start = True
                safety_pol_loss = 0  
                for i in range(self.safety_training_epoch):
                    safety_policy_loss = self.safety_agent.update_parameters(self.safety_constrained_memory, self.cfg.batch_size)
                    safety_pol_loss+=safety_policy_loss
                    
                safety_pol_loss = safety_pol_loss/self.safety_training_epoch
                
                safety_pol_loss_vec.append(safety_pol_loss)
                # rec_critic_loss_vec.append(rec_crtc_loss)
          
            if (n>0) and (n%2==0):
                eval_reward, eval_cost, eval_step  = self.run_safety_eval_episode()
                # max_env = self.cfg.max_episode_steps
                safety_policy_reward.append(eval_reward)
                safety_policy_cost.append(eval_cost)
                 #------------------------------------------------
                if self.best_cost>=eval_cost:
                    self.best_cost = eval_cost
                    self.safety_agent.save_best_safety_model(self.best_cost)
                if (n%10)==0:
                    self.safety_agent.save_best_safety_model(eval_cost, interval=n)
               
                print("-"*10,"Evaluation","-"*10)
                print(f'Episode: {n}  '
                      f'Cost: {eval_cost:<6.2f}  '
                      f'Reward: {eval_reward:<6.2f}  '
                      f'Episode Steps: {eval_step:<6.2f}  '
                )
                print("-"*(20+len(" Evaluation ")))
                #----------------Plot Loss data-----------------------
                plot_dir= os.path.join(self.logdir, 'Plots')
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                #----------------Plot data-----------------------
                safety_pl_plot_dir= os.path.join(plot_dir, 'safety_policy_loss_plot')
                self.plot(safety_pol_loss_vec, safety_pl_plot_dir, loss=True)

                safety_plot_dir= os.path.join(plot_dir, 'safety_plot')
                self.plot(safety_policy_cost, safety_plot_dir, safety=True)

            # if n%20==0:
                # sfty_reward, sfty_cost, tsk_reward, tsk_cost = self.run_task_rec_episode()
                # task_safety_reward.append(sfty_reward)
                # task_safety_cost.append(sfty_cost)
                # only_tsk_reward_vec.append(tsk_reward)
                # only_tsk_cost_vec.append(tsk_cost)
                # plot_dir= os.path.join(self.logdir, 'Plots')
                # if not os.path.exists(plot_dir):
                #     os.makedirs(plot_dir)
                # task_rec_plot_dir= os.path.join(plot_dir, 'task_rec_reward_plot')
                # self.plot_task_safety_reward(task_safety_reward, task_safety_cost, only_tsk_reward_vec, only_tsk_cost_vec, task_rec_plot_dir)
                #------------------------------------------------
    
    def run_safety_train_episode(self, warmup):
        #****************************************
        safety_reward, safety_cost, safety_step = self.rollout_trajectory(warmup=warmup)
        return safety_reward, safety_cost, safety_step
        #****************************************
        
        #****************************************
    def select_random_action(self, env):
        action = np.random.uniform(env.action_space.low.min(), 
                                   env.action_space.high.max(),
                                   env.action_space.shape).astype(np.float32)
        return action

  #====================================================================
    def rollout_trajectory(self, warmup=False, safety_iter=1):
        if warmup:
          print("Warm up rollouts")
        env = gym.make(self.cfg.configure_env)
        env.seed(111)

        reward_vec = []
        cost_vec = []
        step_vec = []

        reward_total = 0
        cost_total = 0
        total_safety_steps = 0

        #----------------------
        episode_steps = 0
        state = env.reset()
        done = False
        rollout_cost = 0
        while not done:
            episode_steps+=1
            ##############################################
            tmp_env = copy.deepcopy(env)
            layout = env.get_layout()
            copy_world = env.get_world()
            ##############################################
            t_r = 0
            t_c = 0
            ep_stps = 0
            for _ in range(safety_iter):
                 t_r ,t_c, ep_stps = self.safety_trajectory(tmp_env, copy_world, layout, state, warmup=warmup)
                 reward_total+=t_r
                 cost_total+=t_c
                 total_safety_steps+=ep_stps
            reward_total = reward_total/safety_iter
            cost_total = cost_total/safety_iter
            total_safety_steps = total_safety_steps/safety_iter
            # print("Rollout step:", episode_steps)
            reward_vec.append(reward_total)
            cost_vec.append(cost_total)
            step_vec.append(total_safety_steps)
            reward_total = 0
            cost_total = 0
            total_safety_steps = 0 
            #-------------------------------------------------------------
            if np.random.rand()<self.beta:
                action = self.adv_agent.select_action(state)
            else:
                if np.random.rand()>self.eta:
                    action = self.expert_agent.select_action(state)
                else:
                    action = env.action_space.sample()
            #-------------------------------------------------------------
            next_state, _, done, info = env.step(action)
            state = next_state
            if "cost" in info:
                cost = info["cost"]
                rollout_cost += float(cost)
            done = True if episode_steps==self.cfg.max_episode_steps or rollout_cost>100 else done
            done = True if "goal_met" in info and info["goal_met"] else done
            if done:
                # print("Terminate rollout")
                break
        # print("Return to training loop")
        return np.average(reward_vec), np.average(cost_vec), np.average(step_vec)
  #==============================================================


    def safety_trajectory(self, copied_env, mj_sim_world, layout, init_state, warmup=False, deadlock=False, capacity=5):
        ############ SAFETY GYM ####################
        # print("Safety trajectory called")
        sfty_env =  gym.make(self.cfg.configure_env)
        sfty_env.seed(111)
        ############################################
        # FORCE RESET TO INITIAL STATE AS ROLLOUT
        _ = sfty_env.reset()
        sfty_env.set_layout(layout)
        sfty_env.set_world(mj_sim_world)
        ############################################
        state = init_state
        total_cost = 0
        total_reward=0
        epi_steps = 0
        done = False
        ###########################################
         # PENALIZE DEAD LOCK OR STUCK STATE
        if deadlock:
            deadlock_detection_vec = []
            buffer_cap = capacity
            buff_index = 0
            no_of_cycle = 0
            deadlock_penalty = 1
        ###########################################
        while not done:
            epi_steps+=1
            ##############################################
            if deadlock:
                if len(deadlock_detection_vec) < buffer_cap:
                      deadlock_detection_vec.append(None)
                deadlock_detection_vec[buff_index]=(list(state))
                buff_index = (buff_index+1)%buffer_cap
            ##############################################
            if warmup:
                action = sfty_env.action_space.sample()
            else:
                action = self.safety_agent.select_action(state, eval=False)
            # print(action)
            next_state, reward, done, info = sfty_env.step(action)
            total_reward+=reward
            if "cost" in info:
                cost = info["cost"]
                total_cost += float(cost)
            mask = float(not done)
            ######### DEAD LOCK DETECTION ##################
            for item in deadlock_detection_vec:
                if (item-list(next_state))==0:
                    ##### DEADLOCK ##########
                    cost +=deadlock_penalty
            ################################################
            self.safety_constrained_memory.push(state, action, reward, cost, next_state, mask)
            state = next_state
            done = True if epi_steps-1==self.cfg.max_episode_steps else done
            done = True if "goal_met" in info and info["goal_met"] else done
            if done:
                sfty_env.close()
                break
        return total_reward, total_cost, epi_steps
#==============================================================

    def run_safety_eval_episode(self):
        total_cost = 0
        total_reward = 0
        total_step = 0
        episode_steps = 0
        state = self.test_env.reset()
        done = False
        while not done:
            episode_steps+=1
            action = self.safety_agent.select_action(state, eval=True)
            next_state, reward, done, info = self.test_env.step(action)
            total_reward+=reward
            if "cost" in info:
                cost = info["cost"]
                total_cost += float(cost)

            state = next_state
            done = True if episode_steps-1==self.cfg.max_episode_steps else done
            done = True if "goal_met" in info and info["goal_met"] else done
            if done:
                break
        return total_reward, total_cost, episode_steps
  #====================================================================
  #====================================================================
    def torchify(self, x): 
        return torch.FloatTensor(x).to(self.device).unsqueeze(0)

    def run_task_rec_episode(self):
        total_cost = 0
        total_reward = 0
        step_cnt = 0
        tsk_cnt = 0

        done = False
        state = self.test_env.reset()
        #########################################
        tmp_env = copy.deepcopy(self.test_env)
        copy_world = self.test_env.get_world()
        ########################################
        epi_task_reward, epi_task_cost = self.run_task_only_episode(tmp_env, copy_world, state)
        epi_step_count=0
        while not done:
            epi_step_count+=1
            action_tsk = self.expert_agent.select_action(state)
            shield_val_tsk = self.safety_agent.get_shield_value(self.torchify(state), self.torchify(action_tsk))
            if shield_val_tsk>self.shield_theshold:   
                action = self.safety_agent.select_action(state, eval=True)
                safety_cnt+=1
            else:
                action = action_tsk
                tsk_cnt+=1
            n_state, reward, done, info = self.test_env.step(action)
            total_reward += reward
            if "cost" in info:
                cost = info["cost"]
                total_cost += float(cost)
            state = n_state
            done = done or epi_step_count==self.test_env._max_episode_steps
            if done:
                break
        return total_reward, total_cost, epi_task_reward, epi_task_cost
  #====================================================================
    def run_task_only_episode(self, copied_env, mj_sim_world, init_state):
        ############ SAFETY GYM ####################
        tsk_env = gym.make(self.cfg.configure_env)
        tsk_env.seed(111)
        _ = tsk_env.reset()
        tsk_env.layout = copied_env.layout
        tsk_env.set_world(mj_sim_world)
        ############################################
        state = init_state
        done =False
        total_reward = 0
        total_cost = 0
        epi_step_count=0
        while not done:
            epi_step_count+=1
            action = self.expert_agent.select_action(state)
            nxt_state, reward, done, info = tsk_env.step(action)
            total_reward+=reward
            if "cost" in info:
                cost = info["cost"]
                total_cost += float(cost)
            state = nxt_state
            done = True if epi_step_count==self.cfg.max_episode_steps else done
            done = True if "goal_met" in info and info["goal_met"] else done
            if done:
                break
        return total_reward, total_cost
  #====================================================================
            
    def plot(self, data1, path, loss=False, safety=False):
        plt.figure()
        plt.plot(range(len(data1)), data1)
        if loss:
            plt.xlabel('No of iter')
            plt.ylabel('Loss:')
        if safety:
            plt.xlabel('No of iter:')
            plt.ylabel('Cost:')
        plt.savefig(path+'.png', format='png')
        plt.close()

    def plot_task_safety_reward(self, sfty_reward, sfty_cost, tsk_reward, tsk_cost,  path):
        plt.figure()
        plt.plot(range(len(sfty_reward)), sfty_reward, "-g", linewidth=3.0 ,label="Task+Shield+Safety reward")
        plt.plot(range(len(sfty_cost)), sfty_cost, ":g", linewidth=2.0 ,label="Task+Shield+Safety cost",)
        plt.plot(range(len(tsk_reward)), tsk_reward, "-b",label="task policy reward")
        plt.plot(range(len(tsk_cost)), tsk_cost, ":b",label="task policy cost")
        plt.legend(loc="best")
        plt.savefig(path+'.png', format='png')
        plt.close()


            






            
        
        
