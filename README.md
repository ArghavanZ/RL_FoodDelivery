# RL_FoodDelivery 🚁🍱

This repository contains the codebase for the research paper "Multi-modal Meal Delivery Dynamic Pricing Using Deep Reinforcement Learning".



This research exploring **dynamic pricing** for food delivery platforms using **drones and ground vehicles**. A reinforcement learning (RL) approach is employed to optimize decisions for a heterogeneous population of customers.

## 📌 Project Objectives

- Model a realistic urban delivery environment with drones and vehicles
- Develop a custom RL environment using Gymnasium Package
- Train deep reinforcement learning agents to optimize delivery pricing or routing using PPO. 
- Compare RL performance against heuristic baseline policies


## 🏗️ Repository Structure

```bash
drone-food-delivery/
├── env_config/ # YAML Environment Parameters
├── eval/ # Evaluation results
├── hyperparameters/ # YAML Hyperparameters for RL algorithm
├── network/data # network data 
├── scripts/ # Training and evaluation scripts
├── src/ # Source code for environments, and utils
├── results/ # Trained agents
├── .gitignore
├── LICENSE
├── README.md # Project overview and instructions
└── requirements.txt # Python dependencies
 ```


## I. Installation
 
1. Use conda or your favourite package manager.

    ```bash
    conda create -n drone_rl python=3.11
    conda activate drone_rl
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    pip install -r requirements.txt
    ```
    NOTE:
    Make sure to save the path where *src/* is located every time you start a new terminal. Most scripts will contain code at the top like:

    ```python
    import os, sys
    ROOT = os.path.abspath(os.curdir)
    sys.path.append(os.path.abspath(os.path.join(ROOT,'src')))
    ```

    This handles the issue, but it can also be resolved directly as such:

    ```bash
    export PYTHONPATH="$PWD/src"
    ```

## II. Running the RL algorithm on environment and evaluate it

First to train the RL algorithm, use one of the scenarios:

Scenario1:
```bash
conda activate drone_rl
python scripts/train1.py --start_seed s --n_seeds l --HP_path hyperparameters/H1.yaml --C_path env_config/env_c10.yaml 
```

Scenario2:
```bash
conda activate drone_rl
python scripts/train2.py --start_seed s --n_seeds l --HP_path hyperparameters/H1.yaml --C_path env_config/env_c20.yaml 
```

To evaluate each scenario use the following code:

Scenario1:
```bash
conda activate drone_rl
python scripts/eval1.py --save_path eval/eval_env_c10_H1_r0/eval.txt --C_path env_config/env_c10.yaml --model_dir results/drone_1/PPO/env_c10_H1_run_0/models/model.zip --run_name eval_env_c10_H1_r0
```

Scenario2:
```bash
conda activate drone_rl
python scripts/eval2.py --save_path eval/eval_env_c20_H1_r0/eval.txt --C_path env_config/env_c20.yaml --model_dir results/drone_2/PPO/env_c20_H1_run_0/models/model.zip --run_name eval_env_c20_H1_r0
```


Note that:
1. In file *env_config/*, each config file is named by env_c followed by the config id which its first digit indicates the scenario: *env_c##/*

2. To run the long trains, we use our server zapdos which support tmux. This step should be done before starting any training process.

3. In file *hyperparameters/*, each file is named by the the version number : *H#.yaml/*
    - These are the RL Algorithm hyperparameters

```bash
tmux new -s mysession
Ctrl + b then d #(detached from current session it will go back to bash)
tmux attach -t mysession #(attach again)
```


## III. References 
#### gymnasium
- [Documentation](https://gymnasium.farama.org/)
- [Creating a custom env (FULL)](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py)
#### stablebaselines3
- [Documentation](https://stable-baselines3.readthedocs.io/en/master/)
#### wandb
- [Quickstart](https://docs.wandb.ai/quickstart/)



