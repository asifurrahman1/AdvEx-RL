import argparse
def CarGoal1_config():
    parser = argparse.ArgumentParser(description='AdvEx-RL RL Safety Arguments')
    parser.add_argument('--configure-env', default='none', help='')
    parser.add_argument('--env-change', type=float, default=1.0, help='multiplier for variation of env dynamics')
    parser.add_argument('--max-episode-steps', type=int, default=1000, help='Set maximum number of step before termination')
    parser.add_argument('--env-name', default='none', help='Gym environment (default: maze)')
    parser.add_argument('--exp-data-dir', default='/Experimental_Data/', help='Set experiment data location')
    parser.add_argument('--device', default='', help='run on CUDA (default: False)')
    parser.add_argument('--logdir_suffix', default='', help='log directory suffixy')
    parser.add_argument('--epoch', type=int, default=1, help='model updates per simulator step (default: 1)')   #Nav 1 (1)
    parser.add_argument('--seed', type=int, default=123456, help='random seed (default: 123456)')
    parser.add_argument('--train_start', type=int, default=2, help='No of episode to start training')
    parser.add_argument('--num_steps', type=int, default=1000000, help='maximum number of steps (default: 1000000)')
    parser.add_argument('--num_eps', type=int, default=1000000, help='maximum number of episodes (default: 1000000)')
    parser.add_argument('--model_path', default='runs', help='exterior log directory')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size (default: 256)')
    parser.add_argument('--saved_model_path', default='/Trained_models/TASRL/Safexp-CarGoal1-v0/TASRL_at_interval_140_safety_0.0', help='exterior log directory')
    #=========================================================================================================
    ######################################################################################################
    #=========================================================================================================
    parser.add_argument('--shield-threshold', type=float, default=0.99, help='learning rate (default: 0.0003)')
    parser.add_argument('--adv_path_shield', default='/Trained_models/SAC/Safexp-CarGoal1-v0/Adversary/13500_Agent_Model_808.0', help='Specify trained adversary policy path')
    parser.add_argument('--task_path', default='/Trained_models/TRPO/Safexp-CarGoal1-v0/Taskpolicy/2023-01-13_19-06-59/best_model_27.78041160440346', help='Specify trained task policy path')
    parser.add_argument('--adv_path', default='/Trained_models/TRPO/Safexp-CarGoal1-v0/Adversary/2023-01-13_21-36-57/best_model_805.4', help='Specify trained adversary policy path')
    #=========================================================================================================
    # parser.add_argument('--safety_path', default='/Trained_models/AdvExRL/Safexp-CarGoal1-v0/AdvExRL_at_interval_240_safety_0.0', help='Specify trained task policy path')
    # parser.add_argument('--safety_path', default='/Trained_models/AdvExRL/Safexp-CarGoal1-v0/Feb-10-2023_Best_safety_ratio0.0', help='Specify trained task policy path')
    parser.add_argument('--safety_path', default='/Trained_models/TASRL/Safexp-CarGoal1-v0/TASRL_at_interval_140_safety_0.0', help='Specify trained task policy path')
    ######################################################################################################
    #=========================================================================================================
    parser.add_argument('--logdir', default='runs', help='exterior log directory')
    parser.add_argument('--gamma',type=float,default=0.99,help='discount factor for reward (default: 0.99)')
    parser.add_argument( '--tau',type=float,default=0.005, help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha',type=float, default=0.20, help='Temperature parameter α determines the relative importance of the entropy\
                                                                  term against the reward (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate (default: 0.0003)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 256)')
    parser.add_argument('--policy', default='Gaussian', help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, help='size of replay buffer (default: 1000000)')
    #=================================================================================
    #=================================================================================
    parser.add_argument('--beta', type=float, default=0.7, help='Rollout agent - adversarial sample ratio default 0.7')
    parser.add_argument('--eta', type=float, default=0.5, help='Rollout agent - expert sample ration default 0.5*(1-adversarial sample ratio)')
    #=================================================================================
    return parser.parse_args()

def CarButton1_config():
    parser = argparse.ArgumentParser(description='AdvEx-RL RL Safety Arguments')
    parser.add_argument('--configure-env', default='none', help='')
    parser.add_argument('--env-change', type=float, default=1.0, help='multiplier for variation of env dynamics')
    parser.add_argument('--max-episode-steps', type=int, default=400, help='Set maximum number of step before termination')
    parser.add_argument('--env-name', default='none', help='Gym environment (default: maze)')
    parser.add_argument('--exp-data-dir', default='/Experimental_Data/', help='Set experiment data location')
    parser.add_argument('--device', default='', help='run on CUDA (default: False)')
    parser.add_argument('--logdir_suffix', default='', help='log directory suffixy')
    parser.add_argument('--epoch', type=int, default=1, help='model updates per simulator step (default: 1)')   #Nav 1 (1)
    parser.add_argument('--seed', type=int, default=123456, help='random seed (default: 123456)')
    parser.add_argument('--train_start', type=int, default=2, help='No of episode to start training')
    parser.add_argument('--num_steps', type=int, default=1000000, help='maximum number of steps (default: 1000000)')
    parser.add_argument('--num_eps', type=int, default=1000000, help='maximum number of episodes (default: 1000000)')
    parser.add_argument('--model_path', default='runs', help='exterior log directory')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size (default: 256)')
    parser.add_argument('--saved_model_path', default='/Trained_models/TASRL/Safexp-CarButton1-v0/TASRL_at_interval_210_safety_0.0', help='exterior log directory')
    #=========================================================================================================
    ######################################################################################################
    #=========================================================================================================
    parser.add_argument('--shield-threshold', type=float, default=0.99, help='learning rate (default: 0.0003)')
    parser.add_argument('--adv_path_shield', default='/Trained_models/SAC/Safexp-CarButton1-v0/Adversary/DateTime_Jan-21-2023_reward_792.0', help='Specify trained adversary policy path')
    parser.add_argument('--task_path', default='/Trained_models/TRPO/Safexp-CarButton1-v0/Taskpolicy/2023-01-21_20-23-00/best_model_26.124586376394454', help='Specify trained task policy path')
    parser.add_argument('--adv_path', default='/Trained_models/TRPO/Safexp-CarButton1-v0/Adversary/2023-01-21_20-23-27/best_model_695.4', help='Specify trained adversary policy path')
    #=========================================================================================================
    parser.add_argument('--safety_path', default='/Trained_models/TASRL/Safexp-CarButton1-v0/TASRL_at_interval_210_safety_0.0', help='Specify trained task policy path')
    ######################################################################################################
    #=========================================================================================================
    parser.add_argument('--logdir', default='runs', help='exterior log directory')
    parser.add_argument('--gamma',type=float,default=0.99,help='discount factor for reward (default: 0.99)')
    parser.add_argument( '--tau',type=float,default=0.005, help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha',type=float, default=0.20, help='Temperature parameter α determines the relative importance of the entropy\
                                                                  term against the reward (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate (default: 0.0003)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 256)')
    parser.add_argument('--policy', default='Gaussian', help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, help='size of replay buffer (default: 1000000)')
    #=================================================================================
    #=================================================================================
    parser.add_argument('--beta', type=float, default=0.7, help='Rollout agent - adversarial sample ratio default 0.7')
    parser.add_argument('--eta', type=float, default=0.5, help='Rollout agent - expert sample ration default 0.5*(1-adversarial sample ratio)')
    #=================================================================================
    return parser.parse_args()


def get_safety_args(env_name):
    if env_name == "Safexp-CarGoal1-v0":
      return CarGoal1_config()
    elif env_name == "Safexp-CarButton1-v0":
      return CarButton1_config()
    