import torch
import gym
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
from matplotlib import pyplot 
import pickle
import matplotlib.pyplot as plt
import warnings
import argparse
from AdvExRL_safetygym.trpo_eval_agent import TRPO
from AdvExRL_safetygym.safety_policy import Safety_Agent
from AdvExRL_safetygym.safety_config import get_safety_args
from sac_agent.sac import SAC 
from sac_agent.adversary_config import get_adv_args
from pytorch_trpo.running_state import ZFilter
from tqdm import tqdm
from matplotlib import pyplot 
import matplotlib.pyplot as plt
import pickle
warnings.filterwarnings("ignore")

try:
    import safety_gym.envs
except ImportError:
    print("can not find safety gym...")

# def torchify(x, device=None):
#     if device==None:
#         device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
#     else:
#         device=device
#     return torch.FloatTensor(x).to(device).unsqueeze(0)
def use_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def plot_histogram(data, path, num=''):
      plt.figure()
      plt.hist(data, 
              20,
              density=True,
              histtype='bar',
              facecolor='b',
              alpha=0.5
              )
      # plt.hist(data, density=False, bins=20)
      plt.ylabel('Frequency')
      plt.xlabel('Critic Value')
      plt.savefig(path+'/critc_val_hist{}.png'.format(num), format='png')
      plt.close()

def plot_return(shield_threshold, return_data_vec, path, ext=''):
      fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
      plt.rcParams.update({'font.size': 25})
      ax.fill_between(shield_threshold, return_data_vec, 0, color='skyblue', 
                                alpha=.25, label='')
      ax.plot(shield_threshold, return_data_vec, color='skyblue', linewidth='3')
      ax.set_xticks(shield_threshold)
      ax.tick_params(axis='x', labelsize=20)
      ax.tick_params(axis='y', labelsize=20)
      ax.set_xlabel('Safety threshold value', fontsize=30)
      ax.set_ylabel('{}'.format(ext), fontsize=30)
      plt.savefig(path+'/sensitivity{}_wrt_different_threshold.png'.format(ext),dpi=300,format='png',bbox_inches='tight')


def eval_critic_val_episode(test_env, 
                            task_agent, 
                            adv_agent=None,
                            adv_shield=None,
                            safety_policy=None,
                            m_episode_steps=1000,
                            use_random=False,
                            cap=500,
                            safety_shield=False
                            ):
    num_steps = 0
    reward_sum = 0
    cost_sum = 0
    critic_vec = []
    indx = 0
    capacity = cap
    ########################
    state = test_env.reset()
    done = False
    while not done:
        num_steps+=1
        if not use_random:
            action = task_agent.select_action(state)
        else:
            action = test_env.action_space.sample()
        #-----------------------------------
        # shield_val = shield_val.cpu().detach().numpy()[0]
        next_state, reward, done, info = test_env.step(action)
        if "cost" in info:
            cost = info["cost"]
        if cost>0:
            if safety_shield:
                shield_val = safety_policy.get_shield_value(state, action)
            else:
                shield_val = adv_shield.get_shield_value(state, action)
            if len(critic_vec) < capacity:
              critic_vec.append(None)
            critic_vec[indx]=(shield_val)
            indx = (indx+1)%capacity
        #-----------------------------------
        state = next_state
        done = True if num_steps==m_episode_steps else done
        if done:
          #  test_env.close()
           break
    return critic_vec

def eval_shield_cost_sensitivity(test_env, 
                                task_agent, 
                                adv_agent=None, 
                                adv_shield=None,
                                safety_policy=None,
                                m_episode_steps=1000,
                                shield_threshold = 0,
                                cap=5,
                                atk_rate=0.0,
                                safety_shield=False
                                ):
    num_steps = 0
    reward_sum = 0
    cost_sum = 0
    indx = 0
    capacity = cap
    critic_vec = []
    ########################
    state = test_env.reset()
    done = False
    while not done:
        num_steps+=1
        action_tsk = task_agent.select_action(state)
        # action_tsk = test_env.action_space.sample()
        #-----------------------------------
        if np.random.rand()<atk_rate:
            action_tsk = adv_agent.select_action(state)
        #-----------------------------------
        # 
        if safety_shield:
            shield_val = safety_policy.get_shield_value(state, action_tsk)
        else:
            shield_val = adv_shield.get_shield_value(state, action_tsk)

        if shield_val>=shield_threshold:
            action = safety_policy.select_action(state, eval=True)
        else:
            action = action_tsk
        #-----------------------------------
        next_state, reward, done, info = test_env.step(action)
        reward_sum += reward
        if "cost" in info:
            cost = info["cost"]
            cost_sum += float(cost)
        state = next_state
        done = True if num_steps==m_episode_steps else done
        if done:
           break
    return cost_sum, reward_sum

def run(cfg):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_env = gym.make(cfg.env_name)
    test_env.seed(args.seed)
    action_space = test_env.action_space
    observation_shape = test_env.observation_space.shape[0]
    action_shape = test_env.action_space.shape[0]
    max_episode_steps = cfg.max_epi_step
    ##########################################################
    current_path  = os.getcwd()
    # print(current_path)
    time = datetime.now().strftime("%b-%d-%Y|%H:%M|%p")
    config = get_safety_args(cfg.env_name)

    expert_agent_path = current_path+config.task_path
    adv_agent_path = current_path+config.adv_path
    adv_shield_path = current_path+config.adv_path
    safety_policy_path = current_path+config.safety_path
    ###########################################################
    expert_agent_path = current_path+config.task_path
    adv_sheild_path = current_path+config.adv_path_shield
    adv_agent_path = current_path+config.adv_path
    safety_policy_path = current_path+config.safety_path
    m_name = adv_sheild_path.split('Adversary/')[1]
    adv_shield_name = "Adv:"+m_name
    s_name = safety_policy_path.split('{}/'.format(cfg.env_name))[1]
    sfty_model_name = "Safety"+s_name
    model_name = adv_shield_name+":"+sfty_model_name
    result_logdir = os.path.join(config.logdir, cfg.env_name, model_name,'Sensitivity_Tested_at_{}'.format(time))
    ##########################################################
    expert_agent = TRPO(observation_shape = observation_shape,
                        model_dir = expert_agent_path,
                        device = device,
                        action_space=action_space
                        )
    adv_cfg = get_adv_args()  
    adv_shield = SAC(observation_space=observation_shape,
                             action_space = action_shape,
                             args = adv_cfg,
                             device = device,
                             config_path=adv_agent_path
                             )
    adv_shield.load_best_model(adv_sheild_path)
    adv_agent = TRPO(observation_shape = observation_shape,
                     model_dir = adv_agent_path,
                     device = device,
                     action_space=action_space
                    )
    print("Agent Loaded")
    print(max_episode_steps)
    ##########################################################
    safety_agent = Safety_Agent(observation_space = observation_shape, 
                                action_space= action_shape,
                                args=config,
                                logdir=result_logdir,
                                env = test_env,
                                adv_agent=adv_agent,
                                device=device
                                )
    print(safety_policy_path)
    safety_agent.load_safety_model(safety_policy_path)
    print("Safety policy Loaded")
    ##########################################################
    shield_threshold = np.arange(0, 50, 2)
    reward_list_adv_sh = []
    cost_list_adv_sh = []

    reward_list_sfty_sh = []
    cost_list_sfty_sh = []

    critic_vec_tsk_all_adv_shield = []
    critic_vec_rand_all_adv_shield = []
    critic_vec_tsk_all_safety_shield = []
    critic_vec_rand_all_safety_shield = []
    ##########################################################
    for thres in tqdm(shield_threshold):
        reward_sum1 = 0
        cost_sum1 = 0

        reward_sum2 = 0
        cost_sum2 = 0

        critic_value_vector_rand_adv = []
        critic_value_vector_tsk_adv = []
        critic_value_vector_rand_sfty = []
        critic_value_vector_tsk_sfty = []  
        for i in range(cfg.test_epoch):
              c_vec_rand = eval_critic_val_episode(test_env,
                                              expert_agent,
                                              adv_agent=adv_agent,
                                              adv_shield = adv_shield,
                                              safety_policy = safety_agent,
                                              m_episode_steps=max_episode_steps,
                                              use_random=True
                                              )
              c_vec_tsk = eval_critic_val_episode(test_env,
                                                  expert_agent,
                                                  adv_agent=adv_agent,
                                                  adv_shield = adv_shield,
                                                  safety_policy = safety_agent,
                                                  m_episode_steps=max_episode_steps,
                                                  use_random=False
                                                  )
              c_vec_rand2 = eval_critic_val_episode(test_env,
                                              expert_agent,
                                              adv_agent=adv_agent,
                                              adv_shield = adv_shield,
                                              safety_policy = safety_agent,
                                              m_episode_steps=max_episode_steps,
                                              use_random=True,
                                              safety_shield=True
                                              )
              c_vec_tsk2 = eval_critic_val_episode(test_env,
                                                  expert_agent,
                                                  adv_agent=adv_agent,
                                                  adv_shield = adv_shield,
                                                  safety_policy = safety_agent,
                                                  m_episode_steps=max_episode_steps,
                                                  use_random=False,
                                                  safety_shield=True
                                                  )
              critic_value_vector_rand_adv.extend(c_vec_rand)   
              critic_value_vector_tsk_adv.extend(c_vec_tsk)   
              critic_value_vector_rand_sfty.extend(c_vec_rand2)  
              critic_value_vector_tsk_sfty.extend(c_vec_tsk2) 
              cost1, reward1 = eval_shield_cost_sensitivity(test_env,
                                                      expert_agent,
                                                      adv_agent=adv_agent,
                                                      adv_shield = adv_shield,
                                                      safety_policy=safety_agent,
                                                      m_episode_steps=max_episode_steps,
                                                      shield_threshold=thres
                                                      )
              cost_sum1+=cost1
              reward_sum1+=reward1

              cost2, reward2 = eval_shield_cost_sensitivity(test_env,
                                                      expert_agent,
                                                      adv_agent=adv_agent,
                                                      adv_shield = adv_shield,
                                                      safety_policy=safety_agent,
                                                      m_episode_steps=max_episode_steps,
                                                      shield_threshold=thres,
                                                      safety_shield=True
                                                      )
              cost_sum2+=cost2
              reward_sum2+=reward2
        avg_reward1 = float(reward_sum1/cfg.test_epoch)
        avg_cost1 = float(cost_sum1/cfg.test_epoch)
        reward_list_adv_sh.append(avg_reward1)
        cost_list_adv_sh.append(avg_cost1)

        avg_reward2 = float(reward_sum2/cfg.test_epoch)
        avg_cost2 = float(cost_sum2/cfg.test_epoch)
        reward_list_sfty_sh.append(avg_reward2)
        cost_list_sfty_sh.append(avg_cost2)

        critic_hist_path_rand= use_path(result_logdir+'/Plots/rand')
        critic_hist_path_tsk= use_path(result_logdir+'/Plots/tsk')
        # plot_histogram(critic_value_vector_rand_adv,critic_hist_path_rand,str(thres)+'rand_adv')
        # plot_histogram(critic_value_vector_tsk_adv,critic_hist_path_tsk,str(thres)+'task_adv')
        # plot_histogram(critic_value_vector_rand_sfty,critic_hist_path_rand,str(thres)+'rand_sfty')
        # plot_histogram(critic_value_vector_tsk_sfty,critic_hist_path_tsk,str(thres)+'tsk_sfty')
        
        critic_vec_rand_all_adv_shield.extend(critic_value_vector_rand_adv)
        critic_vec_tsk_all_adv_shield.extend(critic_value_vector_tsk_adv)
        critic_vec_tsk_all_safety_shield.extend(critic_value_vector_rand_sfty)
        critic_vec_rand_all_safety_shield.extend(critic_value_vector_tsk_sfty)
    all_critic_hist_path_rand= use_path(result_logdir+'/Plots')
    all_critic_hist_path_tsk= use_path(result_logdir+'/Plots')

    plot_histogram(critic_vec_rand_all_adv_shield,all_critic_hist_path_tsk,'random_adv_sh_ALL')
    plot_histogram(critic_vec_tsk_all_adv_shield,all_critic_hist_path_tsk,'task_adv_sh_ALL')
    plot_histogram(critic_vec_rand_all_safety_shield,all_critic_hist_path_tsk,'random_stfy_sh_ALL')
    plot_histogram(critic_vec_tsk_all_safety_shield,all_critic_hist_path_tsk,'task_sfty_sh_ALL')
    # print(reward_list)
    # print(cost_list)
    sensitivity_path = use_path(result_logdir+'/Plots')
    plot_return(shield_threshold,reward_list_adv_sh,sensitivity_path,ext='Reward_adv_shield')
    plot_return(shield_threshold,cost_list_adv_sh,sensitivity_path,ext='Cost_adv_shield')

    plot_return(shield_threshold,reward_list_sfty_sh,sensitivity_path,ext='Reward_sfty_shield')
    plot_return(shield_threshold,cost_list_sfty_sh,sensitivity_path,ext='Cost_sfty_shield')
    ##########################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='none', help='Set test environment to setup all configuration')
    parser.add_argument('--seed', type=int, default=553, help='Set test environment seed')
    parser.add_argument('--test-epoch', type=int, default=10, help='Set test epoch')
    parser.add_argument('--max-epi-step', type=int, default=400, help='Set max episode step')
    args = parser.parse_args()
    run(args)



