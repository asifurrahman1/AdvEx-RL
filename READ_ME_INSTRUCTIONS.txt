
####################################################################################
                    AdvEx-RL
##################################################################################### 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
             Directory Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
After unzipping the given code zip file, the directory structure should be as follows
-------------------------------------------------------------------------------------   
   Directory                        Information
-------------------------------------------------------------------------------------
- READ_ME_INSTRUCTIONS.txt                       (Contains instruction to run relevant code)
- AdvExRL_MuJoCo_code                     
    - AdvEx_RL                      (Contains all the code files relevant to AdvExRL)
        -adv_trainer.py             (Adversary trainer)
        -memory.py
        -network.py
        -sac.Python         
        -safety_agent.py            (AdvExRL safety policy)
        -safety_trainer.py          (AdvExRL trainer)
        -utils.py
        -victim_trainer.py          (task policy trainer)
    - AdvEx_RL_config               (Contains configuration of aversary, task and safety policy for each environment)
        -adversary_config.py
        -safety_config.py
        -victim_config.py
    - AdvEx_RL_Trained_Models       (Contains trained network parameters for Maze, Navigation 1 and Navigation 2)
        - Adversary                 (Trained adversary's network parameters)
        - Safety_policy             (Trained AdvExRL network parameters)
        - Victim                    (Trained SAC Task policy network parameters)
    - config                        (Contains Environment related configuration)
    - env                           (Contains the Enviroment files)
    - Experimental_Data             (Contains our experimental results and relevant data)
    - plot_scripts                  (Contains python file relevant to plotting the experimental results)
    - RecoveryRL                    (Contains files to conduct testing on the 7 baselines)
        -  RecoveryRL_Model         (Contains 7 trained baselines model parameters. We trained the baselines using the code from "https://github.com/abalakrishna123/recovery-rl")      
            -Maze
                - LR
                - RCPO
                - RP
                - RRL_MF
                - RSPO
                - SQRL
                - unconstrained
            - Navigation1
                -...
            - Navigation2
                -...
        - network.py
        - recoveryRL_args.py
        - recoveryRL_agne.py
        - recoveryRL_models.py
        - recoveryRL_qrisk.py
        - recoveryRL_utils.py
        - recRL_comparison_exp_aaa_atk.py
        - recRL_comparison_exp_random_atk.py
        - render_RecRL_aaa_atk.py
        - render_RecRL_random_atk.py

    - requirements.txt              (Contains the list of dependencies)
    - test_random_atk_experiment.py
    - test_aaa_atk_experiment.py
    - test_random_atk_experiment_changed_dynamics.py
    - test_aaa_atk_experiment_changed_dynamics.python
    - test_ablation_random_atk.py
    - test_ablation_aaa_atk.py
    - deadlock_detection_aaa.py
    - render_episode_with_random_atk.py
    - render_episode_with_aaa_atk.py
    - test_safety_threshold_sensitivity.py
    - train_adv.py
    - train_victim.py
    - train_safety_policy.py
- AdvExRL_SafetyGym_code
    - AdvExRL_safetygym              (Contains all the code files relevant to AdvExRL)
          - safety_config.py
          - buffer_memory.py
          - network.py
          - safety_policy.py
          - safety_trainer.py
          - trpo_eval_agent.py
    - cpo_torch                      (Contains CPO implementation from https://github.com/dobro12/CPO/tree/master/torch)
    - pytorch_trpo                   (Contains TRPO implementation from https://github.com/ikostrikov/pytorch-trpo )
    - plot_scripts                   (Contains python file relevant to plotting the experimental results)
    - sac_agent                      (Contains SAC implementation for Adversary from https://github.com/pranz24/pytorch-soft-actor-critic)
    - safety_gym_file_replace        (Contains files related to SafetyGym Environment for AdvEx-RL safety policy training)
    - test_aaa.py
    - test_random.py
    - test_sensitivity.py
    - train_safety_policy.py

######################################################################
                Requirements
######################################################################
1. Handware and language compiler
- GPU 2 GB (In order to use CUDA)
- RAM 4 GB 
- Python 3.9.7

2. Software/Library requirments
    - Install MuJoCo 
	    - Create a virtual environment using:
            		python3 -m venv ./venv
    	    - Activate the virtual environment:
            		source venv/bin/activate
            - Get the mujoco200 ROM from http://www.roboti.us/
            - Specify the path to mujoco200 bin folder [ROM-bin-Path]
                export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:[ROM-bin-Path]
            - Install mujoco 2.0.2.13
                pip install mujoco-py 2.0.2.13

    - Go inside the folder AdvExRL_MuJoCo_code    
        - Install other dependency using the following command
                    pip install -r requirements.txt
        - Download the trained models' for AdvEx-RL and Task policy from "https://drive.google.com/file/d/11Wl1ddfxR5tExoHm4kFtYoCvoqL2GsYx/view?usp=sharing"
            - Extract the zip folder inside AdvExRL_MuJoCo_code (Path should be AdvExRL_MuJoCo_code/AdvEx_RL_Trained_Models)
        - Download the trained models' for the baselines from "https://drive.google.com/file/d/1hRcV-QgrTY7PQ9HfEXDF1qeFWhHJX9vF/view?usp=sharing"
            - Extract the zip folder inside AdvExRL_MuJoCo_code/RecoveryRL (Path should be AdvExRL_MuJoCo_code/RecoveryRL/RecoveryRL_Model)
    
    - Go inside the folder AdvExRL_SafetyGym_code
        - Install other dependency using the following command
                pip install -r requirements.txt
        - Install safety-gym from https://github.com/openai/safety-gym
                - After installation, inside safety-gym folder go to "safety-gym/safety_gym/envs"
                - Replace the "engine.py" and "world.py" file with the files given inside "/AdvExRL_SafetyGym_code/safety_gym_file_replace"
                  (This is done to enable AdvEx-RL safety policy training in a demo rollout mechanism)

                install safety-gym: pip install -e .
        - Download the trained models' from "https://drive.google.com/file/d/1zinU4I99_vhUCnsOV0B9pwe7BnLd-D7t/view?usp=sharing"
            - Extract the downloaded zip folder inside AdvExRL_SafetyGym_code (Path should be AdvExRL_SafetyGym_code/Trained_models)
######################################################################
                  Experiments
######################################################################
1. MUJOCO ENVIRONMENT FROM https://github.com/abalakrishna123/recovery-rl
######################################################################
---------------------------------------------------------------------
            Load pretrained models to reproduce the results 
---------------------------------------------------------------------
Information about the command line arguments:

# Environment specific model configuration can be initialized by passing 
the name of the environment to the argument,

--configure-env [maze] or [nav1] or [nav2]

# Experimental result's save directory can be specified using the argument,
--exp-data-dir [directory]       


Use the following commands to run corresponding comparative experiments 
with pretrained parameters given in folder "AdvEx_RL_Trained_Models" 
and "RecoveryRL_Model" (Contains trained parameters of the baselines) 

******************************************************************************************************************************
Experiment 1: Comparative test under random action perturbation with same environment dynamics as testing environment
******************************************************************************************************************************
    (1.a) To conduct experiment on maze environment run the following command:
          python test_random_atk_experiment.py --configure-env maze --exp-data-dir '/Experimental_Data/Reproduce/Random/maze' 
    
    (1.b) To conduct experiment on Navigation 1 environment run the following command:
          python test_random_atk_experiment.py --configure-env nav1 --exp-data-dir '/Experimental_Data/Reproduce/Random/nav1' 
    
    (1.c) To conduct experiment on Navigation 2 environment run the following command:
          python test_random_atk_experiment.py --configure-env nav2  --exp-data-dir '/Experimental_Data/Reproduce/Random/nav2' 
******************************************************************************************************************************

******************************************************************************************************************************
Experiment 2: Comparative test under AAA perturbation with same environment dynamics as testing environment
******************************************************************************************************************************
    (2.a) To conduct experiment on maze environment run the following command:
          python test_aaa_atk_experiment.py --configure-env maze --exp-data-dir '/Experimental_Data/Reproduce/AAA/maze' 
    
    (2.b) To conduct experiment on Navigation 1 environment run the following command:
          python test_aaa_atk_experiment.py --configure-env nav1 --exp-data-dir '/Experimental_Data/Reproduce/AAA/nav1' 
    
    (2.c) To conduct experiment on Navigation 2 environment run the following command:
          python test_aaa_atk_experiment.py --configure-env nav2 --exp-data-dir '/Experimental_Data/Reproduce/AAA/nav2' 
******************************************************************************************************************************

******************************************************************************************************************************
Experiment 3: Comparative test under random action perturbation with change in environment dynamics
******************************************************************************************************************************
    (3.1.a) To conduct experiment on Navigation 1 by changing the training environment dynamics 5 times run the following command:
            python test_random_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 5.0 --exp-data-dir '/Experimental_Data/Reproduce/Random/Changed_dynamics/nav1/5x'
    
    (3.1.b) To conduct experiment on Navigation 1 by changing the training environment dynamics 10 times run the following command:
            python test_random_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 10.0 --exp-data-dir '/Experimental_Data/Reproduce/Random/Changed_dynamics/nav1/10x'
    
    (3.1.c) To conduct experiment on Navigation 1 by changing the training environment dynamics 15 times run the following command:
            python test_random_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 15.0 --exp-data-dir '/Experimental_Data/Reproduce/Random/Changed_dynamics/nav1/15x'
    
    (3.1.d) To conduct experiment on Navigation 1 by changing the training environment dynamics 20 times run the following command:
            python test_random_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 20.0 --exp-data-dir '/Experimental_Data/Reproduce/Random/Changed_dynamics/nav1/20x'
    
    -----------------------------------------------------------------------------------------------------------------------------------
    (3.2.a) To conduct experiment on Navigation 2 by changing the training environment dynamics 5 times run the following command:
            python test_random_atk_experiment_changed_dynamics.py --configure-env nav2 --env-change 5.0 --exp-data-dir '/Experimental_Data/Reproduce/Random/Changed_dynamics/nav2/5x'
    
    (3.2.b) To conduct experiment on Navigation 1 by changing the training environment dynamics 10 times run the following command:
            python test_random_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 10.0 --exp-data-dir '/Experimental_Data/Reproduce/Random/Changed_dynamics/nav2/10x'
    
    (3.2.c) To conduct experiment on Navigation 1 by changing the training environment dynamics 15 times run the following command:
            python test_random_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 15.0 --exp-data-dir '/Experimental_Data/Reproduce/Random/Changed_dynamics/nav2/15x'
    
    (3.2.d) To conduct experiment on Navigation 1 by changing the training environment dynamics 20 times run the following command:
            python test_random_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 20.0 --exp-data-dir '/Experimental_Data/Reproduce/Random/Changed_dynamics/nav2/20x' 
******************************************************************************************************************************

******************************************************************************************************************************
Experiment 4: Comparative test under AAA perturbation with change in environment dynamics
******************************************************************************************************************************
    (4.1.a) To conduct experiment on Navigation 1 by changing the training environment dynamics 5 times run the following command:
            python test_aaa_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 5.0 --exp-data-dir '/Experimental_Data/Reproduce/Changed_dynamics/AAA/nav1/5x'
    
    (4.1.b) To conduct experiment on Navigation 1 by changing the training environment dynamics 10 times run the following command:
            python test_aaa_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 10.0 --exp-data-dir '/Experimental_Data/Reproduce/Changed_dynamics/AAA/nav1/10x'
    
    (4.1.c) To conduct experiment on Navigation 1 by changing the training environment dynamics 15 times run the following command:
            python test_aaa_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 15.0 --exp-data-dir '/Experimental_Data/Reproduce/Changed_dynamics/AAA/nav1/15x'
    
    (4.1.d) To conduct experiment on Navigation 1 by changing the training environment dynamics 20 times run the following command:
            python test_aaa_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 20.0 --exp-data-dir '/Experimental_Data/Reproduce/Changed_dynamics/AAA/nav1/20x'
    
    -----------------------------------------------------------------------------------------------------------------------------------
    (4.2.a) To conduct experiment on Navigation 2 by changing the training environment dynamics 5 times run the following command:
            python test_aaa_atk_experiment_changed_dynamics.py --configure-env nav2 --env-change 5.0 --exp-data-dir '/Experimental_Data/Reproduce/Changed_dynamics/AAA/nav2/5x'
    
    (4.2.b) To conduct experiment on Navigation 1 by changing the training environment dynamics 10 times run the following command:
            python test_aaa_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 10.0 --exp-data-dir '/Experimental_Data/Reproduce/Changed_dynamics/AAA/nav2/10x'
    
    (4.2.c) To conduct experiment on Navigation 1 by changing the training environment dynamics 15 times run the following command:
            python test_aaa_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 15.0 --exp-data-dir '/Experimental_Data/Reproduce/Changed_dynamics/AAA/nav2/15x'
    
    (4.2.d) To conduct experiment on Navigation 1 by changing the training environment dynamics 20 times run the following command:
            python test_aaa_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 20.0 --exp-data-dir '/Experimental_Data/Reproduce/Changed_dynamics/AAA/nav2/20x' 
******************************************************************************************************************************

******************************************************************************************************************************
Experiment 5: Perform Ablation test under random action perturbation 
******************************************************************************************************************************
    (5.a) To conduct ablation experiment on maze environment run the following command:
          python test_ablation_random_atk.py --configure-env maze --exp-data-dir '/Experimental_Data/Reproduce/Ablation/Random/maze'

    (5.b) To conduct ablation experiment on Navigation 1 environment run the following command:
          python test_ablation_random_atk.py --configure-env nav1 --exp-data-dir '/Experimental_Data/Reproduce/Ablation/Random/nav1'

    (5.c) To conduct ablation experiment on Navigation 2 environment run the following command:
          python test_ablation_random_atk.py --configure-env nav2 --exp-data-dir '/Experimental_Data/Reproduce/Ablation/Random/nav2'

******************************************************************************************************************************

******************************************************************************************************************************
Experiment 6: Perform Ablation test under AAA perturbation 
******************************************************************************************************************************
    (6.a) To conduct ablation experiment on maze environment run the following command:
          python test_ablation_aaa_atk.py --configure-env maze --exp-data-dir '/Experimental_Data/Reproduce/Ablation/AAA/maze'

    (6.b) To conduct ablation experiment on Navigation 1 environment run the following command:
          python test_ablation_aaa_atk.py --configure-env nav1 --exp-data-dir '/Experimental_Data/Reproduce/Ablation/AAA/nav1'

    (6.c) To conduct ablation experiment on Navigation 2 environment run the following command:
          python test_ablation_aaa_atk.py --configure-env nav2 --exp-data-dir '/Experimental_Data/Reproduce/Ablation/AAA/nav2'
******************************************************************************************************************************

******************************************************************************************************************************
Experiment 7: Deadlock detection Test
******************************************************************************************************************************
    (7.a) To conduct deadlock detection experiment on maze environment run the following command:
          python deadlock_detection_aaa.py --configure-env maze

    (7.b) To conduct deadlock detection experiment on Navigation 1 environment run the following command:
          python deadlock_detection_aaa.py --configure-env nav1
    
    (7.c) To conduct deadlock detection experiment on Navigation 2 environment run the following command:
          python deadlock_detection_aaa.py --configure-env nav2

******************************************************************************************************************************
Experiment 8: Safety Threshold Sensitivity Test
******************************************************************************************************************************
    (8.a) To conduct ablation experiment on maze environment run the following command:
          python test_safety_threshold_sensitivity.py --configure-env maze --exp-data-dir '/Experimental_Data/Reproduce/Sensitivity_test/maze'

    (8.b) To conduct ablation experiment on Navigation 1 environment run the following command:
          python test_safety_threshold_sensitivity.py --configure-env nav1 --exp-data-dir '/Experimental_Data/Reproduce/Sensitivity_test/nav1'

    (8.c) To conduct ablation experiment on Navigation 2 environment run the following command:
          python test_safety_threshold_sensitivity.py --configure-env nav2 --exp-data-dir '/Experimental_Data/Reproduce/Sensitivity_test/nav2'
******************************************************************************************************************************

******************************************************************************************************************************
Experiment 9: To Render GIF Execution of Agent under random action perturbation
******************************************************************************************************************************
    (9.a) To conduct ablation experiment on maze environment run the following command:
        python render_episode_with_random_atk.py --configure-env maze --exp-data-dir '/Experimental_Data/Reproduce/RenderGIF/Random/maze'

    (9.b) To conduct ablation experiment on Navigation 1 environment run the following command:
          python render_episode_with_random_atk.py --configure-env nav1 --exp-data-dir '/Experimental_Data/Reproduce/RenderGIF/Random/nav1'

    (9.c) To conduct ablation experiment on Navigation 2 environment run the following command:
          python render_episode_with_random_atk.py --configure-env nav2 --exp-data-dir '/Experimental_Data/Reproduce/RenderGIF/Random/nav2'
******************************************************************************************************************************

******************************************************************************************************************************
Experiment 10: To Render Execution of Agent under AAA perturbation
******************************************************************************************************************************
    (10.a) To conduct ablation experiment on maze environment run the following command:
          python render_episode_with_aaa_atk.py --configure-env maze --exp-data-dir '/Experimental_Data/Reproduce/RenderGIF/AAA/maze'

    (10.b) To conduct ablation experiment on Navigation 1 environment run the following command:
          python render_episode_with_aaa_atk.py --configure-env nav1 --exp-data-dir '/Experimental_Data/Reproduce/RenderGIF/AAA/nav1'

    (10.c) To conduct ablation experiment on Navigation 2 environment run the following command:
          python render_episode_with_aaa_atk.py --configure-env nav2 --exp-data-dir '/Experimental_Data/Reproduce/RenderGIF/AAA/nav2'
******************************************************************************************************************************
---------------------------------------------------------------------
          Train models from Scratch
---------------------------------------------------------------------
---------------------------------------------------------------------
               Training the Baselines
---------------------------------------------------------------------
We trained all the baselines using code from the git repository: "https://github.com/abalakrishna123/recovery-rl"
This repository uses demostration data to train the models, as a result for space limitation we 
are not able to provide the code as supplimentary materials. However after training the 
models using their code we saved the model parameters of the 7 baseline which have been
provided in "/RecoveryRL/RecoveryRL_Model" folder.
---------------------------------------------------------------------

---------------------------------------------------------------------
              AdvEx-RL Training
---------------------------------------------------------------------
1. To change the model configuration of either the task policy, safety policy or 
adversary go to "AdvEx_RL_config" folder and change appropriate hyperparameter
manually

Replace [ENV_NAME] with corresponding environment: maze
                                                   nav1
                                                   nav2
(A) To train the task agent run the following command:
        python train_victim.py --configure-env [ENV_NAME]	
    
    A new directory "AdvEx_RL_Trained_Models_New" will be created 
    in the current root directory and the training models will be saved 
    there inside "Victim" folder

(B) To train the adversary run the following command:
        python train_adv.py	--configure-env [ENV_NAME] 

    A new directory "AdvEx_RL_Trained_Models_New" will be created 
    in the current root directory and the training models will be saved 
    there inside "Adversary" folder   
--------------------------------------------------------------------------------
Before training the AdvExRL Safety_policy, Manually specify the followings in 
the configuration files provided inside the "AdvEx_RL_config" folder: 
        # Set --saved_model_path inside the victim_config.py to the newly trained 
            victim/task agent's model path
        # Set --saved_model_path inside the adversary_config.py to the newly trained 
            adversary's model path
(C) To train the AdvEx-RL safety policy run the following command:
        python train_safety_policy.py --configure-env [ENV_NAME]    

    A new directory "AdvEx_RL_Trained_Models_New" will be created 
    in the current root directory and the training models will be saved 
    there inside "Safety_policy" folder   

    # Before running the experiments (1-9) discussed earlier, using the newly
    trained Safety_policy model parameter, it has to be manually specified in 
    the configuration files provided inside the "AdvEx_RL_config" by:
    
    Changing the --saved_model_path inside the safety_config.py to the newly 
    trained safety_policy's model model path

######################################################################
######################################################################
2. SAFETYGYM ENVIRONMENT FROM https://github.com/openai/safety-gym
#####################################################################
       
******************************************************************************************************************************
Experiment 11: Comparative test under random action perturbation 
******************************************************************************************************************************
   (11.a) To conduct robustness experiment on CarGoal environment run the following command:
          python test_random.py --env-name "Safexp-CarGoal1-v0" --shield 19.8

   (11.b) To conduct robustness experiment on CarButton environment run the following command:
          python test_random.py --env-name "Safexp-CarButton1-v0" --shield 7.9
 
 ******************************************************************************************************************************
Experiment 12: Comparative test under AAA action perturbation 
******************************************************************************************************************************
   (12.a) To conduct robustness experiment on CarGoal environment run the following command:
          python test_aaa.py --env-name "Safexp-CarGoal1-v0" --shield 19.8

   (12.b) To conduct robustness experiment on CarButton environment run the following command:
          python test_aaa.py --env-name "Safexp-CarButton1-v0" --shield 7.9


