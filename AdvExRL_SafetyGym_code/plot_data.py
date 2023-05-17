from matplotlib import pyplot 
import pickle
import matplotlib.pyplot as plt
import warnings
import os
from plot_scripts.plot_util import *


if __name__ == '__main__':
    # result_logdir = '/content/drive/MyDrive/Wake Forest MSc./Research/IJCAI_code_submission/AdvExRL_SafetyGym_code/runs/Safexp-CarGoal1-v0/Adv:13500_Agent_Model_808.0:SafetyAdvExRL_at_interval_140_safety_0.0/AAA_atk_result_sheild_18.9'
    # result_logdir = '/content/drive/MyDrive/Wake Forest MSc./Research/IJCAI_code_submission/AdvExRL_SafetyGym_code/runs/Safexp-CarGoal1-v0/Adv:13500_Agent_Model_808.0:SafetyAdvExRL_at_interval_140_safety_0.0/random_atk_result_sheild_18.9'
    # result_logdir = '/content/drive/MyDrive/Wake Forest MSc./Research/IJCAI_code_submission/AdvExRL_SafetyGym_code/runs/Safexp-CarButton1-v0/Adv:DateTime_Jan-21-2023_reward_792.0:SafetyTASRL_at_interval_210_safety_0.0/AAA_atk_result_sheild_7.9'
    result_logdir = '/content/drive/MyDrive/Wake Forest MSc./Research/IJCAI_code_submission/AdvExRL_SafetyGym_code/runs/Safexp-CarButton1-v0/Adv:DateTime_Jan-21-2023_reward_792.0:SafetyTASRL_at_interval_210_safety_0.0/random_atk_result_sheild_7.9'
    # atk_rate = [0, 0.10, 0.20, 0.30, 0.40, 0.50]
    ####################################################################################
    atk_rate = [0, 0.25, 0.5, 0.75, 1]
    TRPO_data_path = os.path.join(result_logdir,'saved_TRPO_TASRL_Episode_info_data.pkl')
    with open(TRPO_data_path, 'rb') as f1:
        TRPO_vec = pickle.load(f1)
    f1.close()
    print(TRPO_vec)
    CPO_data_path = os.path.join(result_logdir,'saved_CPO_TASRL_Episode_info_data.pkl')
    with open(CPO_data_path, 'rb') as f2:
        CPO_vec = pickle.load(f2)
    f2.close()
    print(CPO_vec)
    ####################################################################################
    TRPO_reward_data_path = os.path.join(result_logdir,'saved_trpo_reward_data.pkl')
    with open(TRPO_reward_data_path, 'rb') as f3:
        trpo_reward_vec_atk = pickle.load(f3)
    f3.close()
    TRPO_cost_data_path = os.path.join(result_logdir,'saved_trpo_cost_data.pkl')
    with open(TRPO_cost_data_path, 'rb') as f4:
        trpo_cost_vec_atk = pickle.load(f4)
    f4.close()
    
    TRPO_AdvExRL_reward_data_path = os.path.join(result_logdir,'saved_TRPO_TASRL_reward_data.pkl')
    with open(TRPO_AdvExRL_reward_data_path, 'rb') as f5:
        TRPO_AdvExRL_reward_safety_vec_atk = pickle.load(f5)
    f5.close()

    TRPO_AdvExRL_cost_data_path = os.path.join(result_logdir,'saved_TRPO_TASRL_cost_data.pkl')
    with open(TRPO_AdvExRL_cost_data_path, 'rb') as f6:
        TRPO_AdvExRL_cost_safety_vec_atk = pickle.load(f6)
    f6.close()

    CPO_reward_data_path = os.path.join(result_logdir,'saved_CPO_reward_data.pkl')
    with open(CPO_reward_data_path, 'rb') as f7:
        CPO_reward_vec_atk = pickle.load(f7)
    f7.close()

    CPO_cost_data_path = os.path.join(result_logdir,'saved_CPO_cost_data.pkl')
    with open(CPO_cost_data_path, 'rb') as f8:
        CPO_cost_vec_atk = pickle.load(f8)
    f8.close()

    CPO_AdvExRL_reward_data_path = os.path.join(result_logdir,'saved_CPO_TASRL_reward_data.pkl')
    with open(CPO_AdvExRL_reward_data_path, 'rb') as f9:
        CPO_AdvExRL_reward_safety_vec_atk = pickle.load(f9)
    f9.close()

    CPO_AdvExRL_cost_data_path = os.path.join(result_logdir,'saved_CPO_TASRL_cost_data.pkl')
    with open(CPO_AdvExRL_cost_data_path, 'rb') as f10:
        CPO_AdvExRL_cost_safety_vec_atk = pickle.load(f10)
    f10.close()

    print("TRPO reward:", trpo_reward_vec_atk)
    print("TRPO+AdvEx-RL:",TRPO_AdvExRL_reward_safety_vec_atk)

    print("CPO reward:", CPO_reward_vec_atk)
    print("CPO+AdvEx-RL reward:",CPO_AdvExRL_reward_safety_vec_atk)

    print("#"*20)
    print("TRPO Cost:", trpo_cost_vec_atk)
    print("TRPO+AdvEx-RL Cost:",TRPO_AdvExRL_cost_safety_vec_atk)

    print("CPO Cost:", CPO_cost_vec_atk)
    print("CPO+AdvEx-RL Cost:",CPO_AdvExRL_cost_safety_vec_atk)
    ####################################################################################
    plot_path = use_path(result_logdir+'/Plots_FINAL/')
    plot_step_count(atk_rate, 
                    TRPO_vec['task_step'] , 
                    TRPO_vec['safety_step'], 
                    CPO_vec['task_step'] , 
                    CPO_vec['safety_step'], 
                    plot_path, 
                    ext='comparison_stepcount_plot')

    plot_reward_result(atk_rate, 
                       trpo_reward_vec_atk, 
                       TRPO_AdvExRL_reward_safety_vec_atk, 
                       CPO_reward_vec_atk, 
                       CPO_AdvExRL_reward_safety_vec_atk,
                       plot_path, 
                       ext='comparison_Reward_plot'
                       )
    plot_cost_result(atk_rate, 
                     trpo_cost_vec_atk, 
                     TRPO_AdvExRL_cost_safety_vec_atk, 
                     CPO_cost_vec_atk,
                     CPO_AdvExRL_cost_safety_vec_atk,
                     plot_path, 
                     ext='comparison_Cost_plot')



    