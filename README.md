# 🧠 Adversarial Behavioral Exploration for Safe Reinforcement Learning (AdvEx-RL)

This repository contains the official implementation of the paper:  
📄 [Adversarial Behavioral Exploration for Safe Reinforcement Learning – IJCAI 2023](https://www.ijcai.org/proceedings/2023/54)

AdvEx-RL introduces a novel adversarial training framework to improve safety and robustness in reinforcement learning by leveraging adversarial exploration and behavior shaping strategies. The framework supports both **MuJoCo** and **Safety Gym** environments and enables extensive evaluation against existing baselines.

---

## 📂 Repository Structure
```
AdvEx-RL/
├── AdvExRL_MuJoCo_code/
│ ├── AdvEx_RL/ # Core AdvEx-RL agent implementation
│ ├── AdvEx_RL_config/ # Configs for adversary, victim, and safety agents
│ ├── AdvEx_RL_Trained_Models/ # Pretrained models
│ ├── RecoveryRL/ # Baseline methods (7 policies)
│ ├── Experimental_Data/ # Output of experiments
│ ├── plot_scripts/ # Plotting scripts
│ ├── env/ # Environment wrappers
│ ├── config/ # Environment configs
│ └── *.py # Evaluation & training scripts
├── AdvExRL_SafetyGym_code/
│ ├── AdvExRL_safetygym/ # AdvEx-RL SafetyGym implementation
│ ├── sac_agent/ # SAC adversary agent
│ ├── cpo_torch/, pytorch_trpo/ # Baseline implementations
│ ├── plot_scripts/
│ ├── safety_gym_file_replace/ # Engine modifications for demo rollout
│ └── *.py
├── requirements.txt
├── LICENSE
└── README.md
```


---

## 📦 Installation Instructions

### ✅ Requirements

- Python 3.9.7  
- GPU with CUDA (≥ 2 GB VRAM recommended)  
- MuJoCo 2.0.2.13  
- RAM ≥ 4 GB  

### 🔧 MuJoCo Setup

1. Download MuJoCo ROM: http://www.roboti.us/
2. Extract and export path:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:[ROM-bin-Path]
```
Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```
Install MuJoCo and dependencies:
```bash
pip install mujoco-py==2.0.2.13
cd AdvExRL_MuJoCo_code
pip install -r requirements.txt
```
Download pretrained models:
- AdvEx-RL Models → extract into AdvEx_RL_Trained_Models

- RecoveryRL Models → extract into RecoveryRL/RecoveryRL_Model

### ⚙️ Safety Gym Setup
Install dependencies:
```bash
cd AdvExRL_SafetyGym_code
pip install -r requirements.txt
```
Install Safety Gym:

```bash
git clone https://github.com/openai/safety-gym
cd safety-gym
pip install -e .
```
Replace Safety Gym engine files:
```bash
cp ../AdvExRL_SafetyGym_code/safety_gym_file_replace/engine.py safety_gym/envs/
cp ../AdvExRL_SafetyGym_code/safety_gym_file_replace/world.py safety_gym/envs/
```
Download SafetyGym trained models → extract into AdvExRL_SafetyGym_code/Trained_models

---

## 🚀 Experiments
All experiments can be reproduced using the pretrained models and the scripts in the repository.

### 🔬 Robustness Experiments
| ID | Experiment Type            | Script                                           | Environment Examples      |
| -- | -------------------------- | ------------------------------------------------ | ------------------------- |
| 1  | Random attack              | `test_random_atk_experiment.py`                  | maze, nav1, nav2          |
| 2  | AAA attack                 | `test_aaa_atk_experiment.py`                     | maze, nav1, nav2          |
| 3  | Random + changed dynamics  | `test_random_atk_experiment_changed_dynamics.py` | --env-change 5.0/10.0/... |
| 4  | AAA + changed dynamics     | `test_aaa_atk_experiment_changed_dynamics.py`    | same as above             |
| 5  | Ablation (Random)          | `test_ablation_random_atk.py`                    | maze, nav1, nav2          |
| 6  | Ablation (AAA)             | `test_ablation_aaa_atk.py`                       | maze, nav1, nav2          |
| 7  | Deadlock Detection         | `deadlock_detection_aaa.py`                      | maze, nav1, nav2          |
| 8  | Threshold Sensitivity      | `test_safety_threshold_sensitivity.py`           | maze, nav1, nav2          |
| 9  | Render Random Perturbation | `render_episode_with_random_atk.py`              | maze, nav1, nav2          |
| 10 | Render AAA Perturbation    | `render_episode_with_aaa_atk.py`                 | maze, nav1, nav2          |


### 🧪 SafetyGym Experiments
| ID | Experiment Type | Script           | Envs                                     |
| -- | --------------- | ---------------- | ---------------------------------------- |
| 11 | Random attack   | `test_random.py` | Safexp-CarGoal1-v0, Safexp-CarButton1-v0 |
| 12 | AAA attack      | `test_aaa.py`    | same                                     |


### 🏋️‍♂️ Train From Scratch
#### Train task agent:
```bash
python train_victim.py --configure-env [maze|nav1|nav2]
```
#### Train adversary:
```bash
python train_adv.py --configure-env [maze|nav1|nav2]
```
Make sure to configure paths inside AdvEx_RL_config to point to the above models.
#### Train AdvEx-RL safety agent:
```bash
python train_safety_policy.py --configure-env [maze|nav1|nav2]
```

---

## 📊 Rendered Demonstrations
AdvEx-RL agent under 100% adversarial attack
<img src="rendered_fig.gif" width="350" height="250"/>

SAC agent under 100% adversarial attack
<img src="SAC_rendered_fig.gif" width="350" height="250"/>

## 📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

## 👤 Author
Md Asifur Rahman
For questions, collaborations, or issues, feel free to open an issue or reach out directly.
