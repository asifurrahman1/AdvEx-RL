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
# from adjustText import adjust_text
from AdvExRL_safetygym.trpo_eval_agent import TRPO
from AdvExRL_safetygym.safety_policy import Safety_Agent
from AdvExRL_safetygym.safety_config import get_safety_args
from pytorch_trpo.running_state import ZFilter
from sac_agent.sac import SAC 
from sac_agent.adversary_config import get_adv_args
from cpo_torch.test_CPO import CPO_TEST
from tqdm import tqdm
from plot_scripts.plot_util import *

warnings.filterwarnings("ignore")

try:
    import safety_gym.envs
except ImportError:
    print("can not find safety gym...")


def eval_episode(test_env, 
                 task_agent, 
                 adv_agent=None, 
                 adv_shield=None,
                 safety_policy=None, 
                 m_episode_steps=1000,
                 atk_rate=0,
                 device=None,
                 random_atk=False,
                 use_safety=False,
                 safety_shield= False,
                 shield_threshold1=17.5,
                 shield_thershold2=None
                 ):
    num_steps = 0
    reward_sum = 0
    cost_sum = 0
    state = test_env.reset()
    done = False

    sfty_cnt=0
    tsk_cnt=0
    while not done:
        num_steps+=1
        action_tsk = task_agent.select_action(state)
        ###################################################
        #************************************************** 
        if np.random.uniform(0,1)<atk_rate and random_atk:
              action_tsk = test_env.action_space.sample()
        #**************************************************   
        ####################################################
        if use_safety:
              if safety_shield:
                  shield_val_tsk = safety_policy.get_shield_value(state, action_tsk)
              else: 
                  shield_val_tsk = adv_agent.get_shield_value(state, action_tsk)
              #*********************************************
              if shield_thershold2==None:
                  if shield_val_tsk>=shield_threshold1:
                    action_sfty = safety_policy.select_action(state, eval=True)
                    shield_val_safety = adv_agent.get_shield_value(state, action_sfty)
                    if shield_val_safety<shield_val_tsk:
                      action=action_sfty
                      sfty_cnt+=1
                    else:
                      action=action_tsk
                      tsk_cnt+=1 
                  else:
                    action = action_tsk 
                    tsk_cnt+=1 
              else:
                  if shield_val_tsk>=shield_threshold1 and shield_val_tsk<shield_thershold2:
                    action_sfty = safety_policy.select_action(state, eval=True)
                    shield_val_safety = adv_agent.get_shield_value(state, action_sfty)
                    if shield_val_safety<shield_val_tsk:
                      action=action_sfty
                      sfty_cnt+=1
                    else:
                      action=action_tsk
                      tsk_cnt+=1 
                  else:
                    action = action_tsk 
                    tsk_cnt+=1  
              #*********************************************
        else:
            action = action_tsk 
            tsk_cnt+=1      
        # #**************************************************   
        ###################################################
        next_state, reward, done, info = test_env.step(action)
        reward_sum += reward
        if "cost" in info:
            cost = info["cost"]
            cost_sum += float(cost)
        state = next_state
        done = True if num_steps==m_episode_steps else done
        if done:
           epi_step_info=dict()
           epi_step_info['epi_step']=num_steps
           epi_step_info['task_count']=tsk_cnt
           epi_step_info['safety_count']=sfty_cnt
           num_steps = 0
           tsk_cnt = 0
           sfty_cnt = 0
           break
    return reward_sum, cost_sum, epi_step_info

def run(cfg):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_env1 = gym.make(cfg.env_name)
    test_env1.seed(args.seed)

    test_env2 = gym.make(cfg.env_name)
    test_env2.seed(args.seed+10)

    test_env3 = gym.make(cfg.env_name)
    test_env3.seed(args.seed+20)

    np.random.seed(args.seed)
    action_space = test_env1.action_space
    observation_shape = test_env1.observation_space.shape[0]
    action_shape = test_env1.action_space.shape[0]
    max_episode_steps = cfg.max_epi_step
    ##########################################################
    shield_thres = cfg.shield
    # shield_upper = 36
    shield_upper = None
    safety_shield = False
    ##########################################################
    current_path  = os.getcwd()
    print(current_path)
    config = get_safety_args(cfg.env_name)
    expert_agent_path = current_path+config.task_path
    adv_sheild_path = current_path+config.adv_path_shield
    adv_agent_path = current_path+config.adv_path
    cpo_agent_path = current_path+"/Trained_models/CPO_models/{}/checkpoint/checkpoint.zip".format(cfg.env_name)
    safety_policy_path = current_path+config.safety_path
    m_name = adv_sheild_path.split('Adversary/')[1]
    adv_shield_name = "Adv:"+m_name
    s_name = safety_policy_path.split('{}/'.format(cfg.env_name))[1]
    sfty_model_name = "Safety"+s_name
    model_name = adv_shield_name+":"+sfty_model_name
    result_logdir = os.path.join(config.logdir, cfg.env_name, model_name, 'random_atk_result_sheild_{}'.format(shield_thres))
    print(result_logdir)
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
    # trpo_state_processor = ZFilter((observation_shape), clip=5)
    print("Agent Loaded")
    print(max_episode_steps)
    ################################################################
    #*************************************************************
    safety_agent = Safety_Agent(observation_space = observation_shape, 
                                action_space= action_shape,
                                args=config,
                                logdir=result_logdir,
                                env = test_env1,
                                adv_agent=adv_shield,
                                device=device
                                )
    safety_agent.load_safety_model(safety_policy_path)
    print("Safety policy Loaded")
    print(cpo_agent_path)
    CPO_test = CPO_TEST(cfg.env_name,
                        max_episode_steps,
                        seed = args.seed+20,
                        adv_agent = adv_agent,
                        adv_shield =adv_shield,
                        safety_policy=safety_agent
                        )
    
    CPO_test.agent.load_cpo_model(cpo_agent_path)
    #*************************************************************
    ################################################################
    atk_rate = [0, 0.25, 0.5, 0.75, 1]
    
    trpo_reward_vec_atk = []
    trpo_cost_vec_atk = []

    cpo_reward_vec_atk = []
    cpo_cost_vec_atk = []

    trpo_TASRL_reward_safety_vec_atk = []
    trpo_TASRL_cost_safety_vec_atk = []

    cpo_TASRL_reward_safety_vec_atk = []
    cpo_TASRL_cost_safety_vec_atk = []

    TRPO_TASRL_episode_steps = []
    TRPO_TASRL_safety_count = []
    TRPO_TASRL_task_count = []

    CPO_TASRL_episode_steps = []
    CPO_TASRL_safety_count = []
    CPO_TASRL_task_count = []
    #**************************************
    for atk in tqdm(atk_rate):
        trpo_reward_sum = []
        trpo_cost_sum = []
        trpo_reward_sum_safety = []
        trpo_cost_sum_safety = []

        cpo_reward_sum = []
        cpo_cost_sum = []
        cpo_reward_sum_safety = []
        cpo_cost_sum_safety = []

        TRPO_TASRL_step_count = 0
        TRPO_TASRL_task_step_count = 0
        TRPO_TASRL_safety_step_count = 0

        CPO_TASRL_step_count = 0
        CPO_TASRL_task_step_count = 0
        CPO_TASRL_safety_step_count = 0
        for i in range(cfg.test_epoch):
              #------------------------------------------------------------------
              trpo_reward, trpo_cost, _  = eval_episode(test_env1,
                                                            expert_agent,
                                                            adv_agent=adv_agent,
                                                            safety_policy=safety_agent,
                                                            m_episode_steps=max_episode_steps,
                                                            atk_rate=atk,
                                                            device=device,
                                                            random_atk=True
                                                            )
              trpo_reward_sum.append(trpo_reward)
              trpo_cost_sum.append(trpo_cost)
              #------------------------------------------------------------------
              #------------------------------------------------------------------
              trpo_reward_sfty, trpo_cost_sfty, info1 = eval_episode(test_env2,
                                                                    expert_agent,
                                                                    adv_agent=adv_agent,
                                                                    safety_policy=safety_agent,
                                                                    m_episode_steps=max_episode_steps,
                                                                    atk_rate=atk,
                                                                    device=device,
                                                                    random_atk=True,
                                                                    use_safety=True,
                                                                    safety_shield= safety_shield,
                                                                    shield_threshold1=shield_thres,
                                                                    shield_thershold2=shield_upper
                                                                    )
              trpo_reward_sum_safety.append(trpo_reward_sfty)
              trpo_cost_sum_safety.append(trpo_cost_sfty)
              #------------------------------------------------------------------
              TRPO_TASRL_step_count += int(info1['epi_step'])
              TRPO_TASRL_task_step_count += float(info1['task_count']/info1['epi_step'])  
              TRPO_TASRL_safety_step_count += float(info1['safety_count']/info1['epi_step'])
              # print(info1['safety_count'])
              #------------------------------------------------------------------
              #------------------------------------------------------------------
              cpo_reward , cpo_cost, _ = CPO_test.random_atk_eval_episode(atk_rate=atk,
                                                                    random_atk= True,
                                                                    use_safety=False,
                                                                    shield_threshold1=shield_thres,
                                                                    shield_thershold2=shield_upper
                                                                    )
              cpo_reward_sum.append(cpo_reward)
              cpo_cost_sum.append(cpo_cost)
              #------------------------------------------------------------------
              #------------------------------------------------------------------
              cpo_reward_sfty , cpo_cost_sfty, info2 = CPO_test.random_atk_eval_episode(atk_rate=atk,
                                                                              random_atk= True,
                                                                              use_safety=True,
                                                                              safety_shield= safety_shield,
                                                                              shield_threshold1=shield_thres,
                                                                              shield_thershold2=shield_upper
                                                                              )
              cpo_reward_sum_safety.append(cpo_reward_sfty)
              cpo_cost_sum_safety.append(cpo_cost_sfty)
              #------------------------------------------------------------------
              CPO_TASRL_step_count += int(info2['epi_step'])
              CPO_TASRL_task_step_count += float(info2['task_count']/info2['epi_step']) 
              CPO_TASRL_safety_step_count += float(info2['safety_count']/info2['epi_step'])
              # print(info2['safety_count'])
              #------------------------------------------------------------------
              
              #**************************************
        trpo_r_mu, trpo_r_lb, trpo_r_ub = get_stats(trpo_reward_sum)
        trpo_c_mu, trpo_c_lb, trpo_c_ub = get_stats(trpo_cost_sum)
        trpo_reward_vec_atk.append([trpo_r_mu, trpo_r_lb, trpo_r_ub])
        trpo_cost_vec_atk.append([trpo_c_mu, trpo_c_lb, trpo_c_ub])
        #**************************************
        trpo_TASRL_r_mu, trpo_TASRL_r_lb, trpo_TASRL_r_ub = get_stats(trpo_reward_sum_safety)
        trpo_TASRL_c_sfty_mu, trpo_TASRL_c_lb, trpo_TASRL_c_ub = get_stats(trpo_cost_sum_safety)
        
        trpo_TASRL_reward_safety_vec_atk.append([trpo_TASRL_r_mu, trpo_TASRL_r_lb, trpo_TASRL_r_ub])
        trpo_TASRL_cost_safety_vec_atk.append([trpo_TASRL_c_sfty_mu, trpo_TASRL_c_lb, trpo_TASRL_c_ub])
        
        TRPO_TASRL_step_count_avg = TRPO_TASRL_step_count/cfg.test_epoch
        TRPO_TASRL_task_step_count_avg = TRPO_TASRL_task_step_count/cfg.test_epoch
        TRPO_TASRL_safety_step_count_avg = TRPO_TASRL_safety_step_count/cfg.test_epoch

        TRPO_TASRL_episode_steps.append(TRPO_TASRL_step_count_avg)
        TRPO_TASRL_task_count.append(TRPO_TASRL_task_step_count_avg)
        TRPO_TASRL_safety_count.append(TRPO_TASRL_safety_step_count_avg) 
        #**************************************
        cpo_r_mu, cpo_r_lb, cpo_r_ub = get_stats(cpo_reward_sum)
        cpo_c_mu, cpo_c_lb, cpo_c_ub = get_stats(cpo_cost_sum)
        cpo_reward_vec_atk.append([cpo_r_mu, cpo_r_lb, cpo_r_ub])
        cpo_cost_vec_atk.append([cpo_c_mu, cpo_c_lb, cpo_c_ub])
        #**************************************
        cpo_TASRL_r_mu, cpo_TASRL_r_lb, cpo_TASRL_r_ub = get_stats(cpo_reward_sum_safety)
        cpo_TASRL_c_sfty_mu, cpo_TASRL_c_lb, cpo_TASRL_c_ub = get_stats(cpo_cost_sum_safety)
        cpo_TASRL_reward_safety_vec_atk.append([cpo_TASRL_r_mu, cpo_TASRL_r_lb, cpo_TASRL_r_ub])
        cpo_TASRL_cost_safety_vec_atk.append([cpo_TASRL_c_sfty_mu, cpo_TASRL_c_lb, cpo_TASRL_c_ub])
    
        CPO_TASRL_step_count_avg = CPO_TASRL_step_count/cfg.test_epoch
        CPO_TASRL_task_step_count_avg = CPO_TASRL_task_step_count/cfg.test_epoch
        CPO_TASRL_safety_step_count_avg = CPO_TASRL_safety_step_count/cfg.test_epoch

        CPO_TASRL_episode_steps.append(CPO_TASRL_step_count_avg)
        CPO_TASRL_task_count.append(CPO_TASRL_task_step_count_avg)
        CPO_TASRL_safety_count.append(CPO_TASRL_safety_step_count_avg)
        
    # **************************************************
    result_logdir=use_path(result_logdir)
    # **************************************************

    TRPO_reward_data_path = os.path.join(result_logdir,'saved_trpo_reward_data.pkl')
    with open(TRPO_reward_data_path, 'wb') as f1:
        pickle.dump(trpo_reward_vec_atk, f1)
    f1.close()
    TRPO_cost_data_path = os.path.join(result_logdir,'saved_trpo_cost_data.pkl')
    with open(TRPO_cost_data_path, 'wb') as f2:
        pickle.dump(trpo_cost_vec_atk, f2)
    f2.close()
    TRPO_AdvExRL_reward_data_path = os.path.join(result_logdir,'saved_TRPO_AdvExRL_reward_data.pkl')
    with open(TRPO_AdvExRL_reward_data_path, 'wb') as f3:
        pickle.dump(trpo_TASRL_reward_safety_vec_atk, f3)
    f3.close()
    TRPO_AdvExRL_cost_data_path = os.path.join(result_logdir,'saved_TRPO_AdvExRL_cost_data.pkl')
    with open(TRPO_AdvExRL_cost_data_path, 'wb') as f4:
        pickle.dump(trpo_TASRL_cost_safety_vec_atk, f4)
    f4.close()

    CPO_reward_data_path = os.path.join(result_logdir,'saved_CPO_reward_data.pkl')
    with open(CPO_reward_data_path, 'wb') as f5:
        pickle.dump(cpo_reward_vec_atk, f5)
    f5.close()

    CPO_cost_data_path = os.path.join(result_logdir,'saved_CPO_cost_data.pkl')
    with open(CPO_cost_data_path, 'wb') as f6:
        pickle.dump(cpo_cost_vec_atk, f6)
    f6.close()

    CPO_TASRL_reward_data_path = os.path.join(result_logdir,'saved_CPO_TASRL_reward_data.pkl')
    with open(CPO_TASRL_reward_data_path, 'wb') as f7:
        pickle.dump(cpo_TASRL_reward_safety_vec_atk, f7)
    f7.close()

    CPO_TASRL_cost_data_path = os.path.join(result_logdir,'saved_CPO_TASRL_cost_data.pkl')
    with open(CPO_TASRL_cost_data_path, 'wb') as f8:
        pickle.dump(cpo_TASRL_cost_safety_vec_atk, f8)
    f8.close()
    ##############################################################
    TRPO_info_dict = dict()
    TRPO_info_dict['epi_steps'] = TRPO_TASRL_episode_steps
    TRPO_info_dict['task_step'] = TRPO_TASRL_task_count
    TRPO_info_dict['safety_step'] = TRPO_TASRL_safety_count
    TRPO_TASRL_info_data_path = os.path.join(result_logdir,'saved_TRPO_TASRL_Episode_info_data.pkl')
    with open(TRPO_TASRL_info_data_path, 'wb') as f9:
        pickle.dump(TRPO_info_dict, f9)
    f9.close()

    CPO_info_dict = dict()
    CPO_info_dict['epi_steps'] = CPO_TASRL_episode_steps
    CPO_info_dict['task_step'] = CPO_TASRL_task_count
    CPO_info_dict['safety_step'] = CPO_TASRL_safety_count
    CPO_TASRL_info_data_path = os.path.join(result_logdir,'saved_CPO_TASRL_Episode_info_data.pkl')
    with open(CPO_TASRL_info_data_path, 'wb') as f10:
        pickle.dump(CPO_info_dict, f10)
    f10.close()
    plot_path = use_path(result_logdir)
    ##############################################################
    plot_reward_result(atk_rate, 
                       trpo_reward_vec_atk, 
                       trpo_TASRL_reward_safety_vec_atk, 
                       cpo_reward_vec_atk, 
                       cpo_TASRL_reward_safety_vec_atk,
                       plot_path, 
                       ext='comparison_Reward_plot'
                       )
    plot_cost_result(atk_rate, 
                     trpo_cost_vec_atk, 
                     trpo_TASRL_cost_safety_vec_atk, 
                     cpo_cost_vec_atk,
                     cpo_TASRL_cost_safety_vec_atk,
                     plot_path, 
                     ext='comparison_Cost_plot')

    plot_step_count(atk_rate,
                    TRPO_TASRL_task_count, 
                    TRPO_TASRL_safety_count, 
                    CPO_TASRL_task_count, 
                    CPO_TASRL_safety_count, 
                    plot_path, 
                    ext='comparison_switch_plot')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='none', help='Set test environment to setup all configuration')
    parser.add_argument('--seed', type=int, default=553, help='Set test environment seed')
    parser.add_argument('--test-epoch', type=int, default=20, help='Set test epoch')
    parser.add_argument('--max-epi-step', type=int, default=4000, help='Set max episode step')
    parser.add_argument('--shield', type=float, default=17.5, help='Set Shield Threshold')
    args = parser.parse_args()
    run(args)



