import os
from matplotlib import pyplot 
import pickle
import matplotlib.pyplot as plt
import numpy as np

def use_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_stats(data):
    if not isinstance(data,(np.ndarray)):
        data = np.array(data)
    size = data.size
    mu = np.mean(data, axis=0)
    lb = mu - np.std(data, axis=0) / np.sqrt(size)
    ub = mu + np.std(data, axis=0) / np.sqrt(size)
    return mu, lb, ub

def get_specific_list(nums, pos):
    result = [i[pos] for i in nums]
    return result
    
def plot_step_count(atk_rate, 
                    trpo_tasrl_task_vec,
                    trpo_tasrl_safety_vec,
                    cpo_tasrl_task_vec,
                    cpo_tasrl_safety_vec,
                    plot_path, 
                    ext=''
                    ):
    fig, ax = plt.subplots(figsize=(10,6), dpi=200)
    #-------------------------------------------------------------
    atk_rate = np.array(atk_rate)*100
    x = atk_rate.copy()
    width = 5

    trpo_tasrl_task_vec = np.array(trpo_tasrl_task_vec)*100
    trpo_tasrl_safety_vec = np.array(trpo_tasrl_safety_vec)*100
    cpo_tasrl_task_vec = np.array(cpo_tasrl_task_vec)*100
    cpo_tasrl_safety_vec = np.array(cpo_tasrl_safety_vec)*100
    print(trpo_tasrl_safety_vec)
    print(cpo_tasrl_safety_vec)
    rects1 = ax.bar(x-1*width, trpo_tasrl_task_vec, width, label='TRPO Task policy',color='blue' )
    rects1 = ax.bar(x-1*width, trpo_tasrl_safety_vec, width, bottom=trpo_tasrl_task_vec, label='TAS-RL Safety policy',color='springgreen' )

    rects2 = ax.bar(x+1*width, cpo_tasrl_task_vec, width, label='CPO Task policy',color='purple' )
    rects2 = ax.bar(x+1*width, cpo_tasrl_safety_vec, width, bottom=cpo_tasrl_task_vec, label='TAS-RL Safety policy',color='springgreen' )

    ax.set_xlabel('Perturbation rate (%)', fontsize=30)
    ax.set_ylabel('Policy percentage (%)', fontsize=30)
    plt.grid(color='gray', linewidth=1, axis='y', alpha=0.5)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=20, fancybox=True, shadow=True, ncol=4)
    plt.savefig(plot_path+'/switch_info{}.png'.format(ext), dpi=300,format='png',bbox_inches='tight')
    plt.show()
    plt.close()


def plot_reward_result(atk_rate,
                       trpo_reward_vec,
                       trpo_tasrl_reward_vec,
                       cpo_reward_vec,
                       cpo_tasrl_reward_vec,
                       path, 
                       ext=''
                       ):
    fig, axs = plt.subplots(figsize=(10,6), dpi=200)
    #-------------------------------------------------------------
    atk_rate = np.array(atk_rate)*100

    trpo_reward_ub = get_specific_list(trpo_reward_vec,2)
    trpo_rewad_lb = get_specific_list(trpo_reward_vec,1)
    trpo_reward_mean = get_specific_list(trpo_reward_vec,0)
    axs.fill_between(atk_rate, trpo_reward_ub, trpo_rewad_lb, color='blue', alpha=.5, label='TRPO Reward')
    axs.plot(atk_rate, trpo_reward_mean, color='blue', linewidth='2')

    trpo_advexrl_reward_sfty_ub = get_specific_list(trpo_tasrl_reward_vec,2)
    trpo_advexrl_reward_sfty_lb = get_specific_list(trpo_tasrl_reward_vec,1)
    trpo_advexrl_reward_sfty_mean = get_specific_list(trpo_tasrl_reward_vec,0)
    axs.fill_between(atk_rate, trpo_advexrl_reward_sfty_ub, trpo_advexrl_reward_sfty_lb, color='cyan', alpha=.5, label='TAS-RL+TRPO Reward')
    axs.plot(atk_rate, trpo_advexrl_reward_sfty_mean, color='cyan', linewidth='2')

    cpo_reward_ub = get_specific_list(cpo_reward_vec,2)
    cpo_rewad_lb = get_specific_list(cpo_reward_vec,1)
    cpo_reward_mean = get_specific_list(cpo_reward_vec,0)
    axs.fill_between(atk_rate, cpo_reward_ub, cpo_rewad_lb, color='magenta', alpha=.5, label='CPO Reward')
    axs.plot(atk_rate, cpo_reward_mean, color='magenta', linewidth='2')

    cpo_advexrl_reward_sfty_ub = get_specific_list(cpo_tasrl_reward_vec,2)
    cpo_advexrl_reward_sfty_lb = get_specific_list(cpo_tasrl_reward_vec,1)
    cpo_advexrl_reward_sfty_mean = get_specific_list(cpo_tasrl_reward_vec,0)
    axs.fill_between(atk_rate, cpo_advexrl_reward_sfty_ub, cpo_advexrl_reward_sfty_lb, color='orange', alpha=.5, label='TAS-RL+CPO Reward')
    axs.plot(atk_rate, cpo_advexrl_reward_sfty_mean, color='orange', linewidth='2')

    labels = ['0', '25', '50', '75']
    axs.tick_params(axis='x', labelsize=20)
    axs.tick_params(axis='y', labelsize=20)

    plt.grid(color='gray', linewidth=1, axis='y', alpha=0.5)
    axs.set_xlabel('Perturbation rate (%)', fontsize=30)
    axs.set_ylabel('Reward', fontsize=30)
    
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fontsize=20, fancybox=True, shadow=True, ncol=2)
    plt.savefig(path+'/TAS_RL_reward_experiment{}.png'.format(ext), dpi=300,format='png',bbox_inches='tight')
    plt.show()
    plt.close()

def plot_cost_result(atk_rate, 
                     cost_vec_atk,
                     trpo_tasrl_cost_vec,
                     cpo_cost_vec,
                     cpo_tasrl_cost_vec,
                     path, 
                     ext=''):
    fig, axs = plt.subplots(figsize=(10,6), dpi=200)
    atk_rate = np.array(atk_rate)*100
    trpo_cost_ub = get_specific_list(cost_vec_atk,2)
    trpo_cost_lb = get_specific_list(cost_vec_atk,1)
    trpo_cost_mean = get_specific_list(cost_vec_atk,0)
    axs.fill_between(atk_rate, trpo_cost_ub, trpo_cost_lb, color='blue', alpha=.5, label='TRPO Cost')
    axs.plot(atk_rate, trpo_cost_mean, color='blue', linewidth='2')

    trpo_advexrl_cost_sfty_ub = get_specific_list(trpo_tasrl_cost_vec,2)
    trpo_advexrl_cost_sfty_lb = get_specific_list(trpo_tasrl_cost_vec,1)
    trpo_advexrl_cost_sfty_mean = get_specific_list(trpo_tasrl_cost_vec,0)
    axs.fill_between(atk_rate, trpo_advexrl_cost_sfty_ub, trpo_advexrl_cost_sfty_lb, color='cyan', alpha=.5, label='TAS-RL+TRPO Cost')
    axs.plot(atk_rate, trpo_advexrl_cost_sfty_mean, color='cyan', linewidth='2')

    cpo_cost_ub = get_specific_list(cpo_cost_vec,2)
    cpo_cost_lb = get_specific_list(cpo_cost_vec,1)
    cpo_cost_mean = get_specific_list(cpo_cost_vec,0)
    axs.fill_between(atk_rate, cpo_cost_ub, cpo_cost_lb, color='magenta', alpha=.5, label='CPO Cost')
    axs.plot(atk_rate, cpo_cost_mean, color='magenta', linewidth='2')

    cpo_advexrl_cost_sfty_ub = get_specific_list(cpo_tasrl_cost_vec,2)
    cpo_advexrl_cost_sfty_lb = get_specific_list(cpo_tasrl_cost_vec,1)
    cpo_advexrl_cost_sfty_mean = get_specific_list(cpo_tasrl_cost_vec,0)
    axs.fill_between(atk_rate, cpo_advexrl_cost_sfty_ub, cpo_advexrl_cost_sfty_lb, color='orange', alpha=.5, label='TAS-RL+CPO Cost')
    axs.plot(atk_rate, cpo_advexrl_cost_sfty_mean, color='orange', linewidth='2')

    labels = ['0', '25', '50', '75']
    axs.tick_params(axis='x', labelsize=20)
    axs.tick_params(axis='y', labelsize=20)
    
    plt.grid(color='gray', linewidth=1, axis='y', alpha=0.5)
    
    axs.set_xlabel('Perturbation rate (%)', fontsize=30)
    axs.set_ylabel('Cost', fontsize=30)
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=20, fancybox=True, shadow=True, ncol=2)
    
    plt.savefig(path+'/TAS_RL_cost_experiment{}.png'.format(ext), dpi=300,format='png',bbox_inches='tight')
    plt.show()
    plt.close()