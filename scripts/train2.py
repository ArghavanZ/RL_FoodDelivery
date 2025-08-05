# set ROOT, and add src directory to path
import os, sys
ROOT = os.path.abspath(os.curdir)
sys.path.append(os.path.abspath(os.path.join(ROOT,'src')))

# import external packages
import wandb

import numpy as np
import pickle
import argparse
import yaml
import torch
import gymnasium as gym
# import stablebaselines
from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

# import internal packages
from RL_FoodDelivery.env.drone_delivery import MMDelivery, make_env
from  RL_FoodDelivery.utils import net

def train(env_cfg, hp_cfg, exp_dir):

    '''
    Main training loop should return model
    '''
    # SETUP ENV
    num_proc = hp_cfg["env_setup"]["num_proc"]  # Number of processes to use
    # fetch env_params from config file
    env_params = env_cfg["env"].copy()
    # Remove unused keys (base params)
    for key in ["name"]:
        env_params.pop(key, None)
    # Convert  etas to numpy arrays 
    env_params["etas"] = np.array(env_params["etas"])
    

    # Create the vectorized environment (expand for readability)
    env_list = [
        make_env(
            params = env_params,
            render_mode=None,
            rank=rank,
            seed=hp_cfg["seed"] * 10,  # Guarantee no repeats
            timelimit=hp_cfg["env_setup"]["timelimit"]
        )
        for rank in range(num_proc)
    ]

    # Pass the list of callables directly to SubprocVecEnv
    env = VecMonitor(SubprocVecEnv(env_list))

     # SETUP MODEL
    algo_name = hp_cfg["model"]["name"]
    policy_kwargs = dict(net_arch=hp_cfg["model"]["net_arch"])

    
    if algo_name == "PPO":
        model = PPO(
            policy=hp_cfg["model"]["policy"], 
            policy_kwargs=policy_kwargs, 
            env=env,
            learning_rate=float(hp_cfg["model"]["learning_rate"]),
            n_steps=(hp_cfg["model"]["n_steps"])//num_proc,
            batch_size=hp_cfg["model"]["batch_size"],
            n_epochs=hp_cfg["model"]["n_epochs"],
            gamma=hp_cfg["model"]["gamma"],
            gae_lambda=hp_cfg["model"]["gae_lambda"],
            clip_range=hp_cfg["model"]["clip_range"],
            ent_coef=hp_cfg["model"]["ent_coef"],
            vf_coef=hp_cfg["model"]["vf_coef"],
            max_grad_norm=hp_cfg["model"]["max_grad_norm"],
            tensorboard_log=f"{exp_dir}/tb",  
            verbose=hp_cfg["model"]["verbose"],
            seed=hp_cfg["seed"],
            device=hp_cfg["device"]
        )
    elif algo_name == "SAC":
        model = SAC(
            policy=hp_cfg["model"]["policy"], 
            policy_kwargs=policy_kwargs, 
            env=env,
            learning_rate=float(hp_cfg["model"]["learning_rate"]),
            batch_size=hp_cfg["model"]["batch_size"],
            buffer_size=hp_cfg["model"]["buffer_size"],
            learning_starts=hp_cfg["model"]["learning_starts"],
            tau=hp_cfg["model"]["tau"],
            gamma=hp_cfg["model"]["gamma"],
            train_freq=hp_cfg["model"]["train_freq"],
            gradient_steps=hp_cfg["model"]["gradient_steps"],
            ent_coef=hp_cfg["model"]["ent_coef"],
            target_update_interval=hp_cfg["model"]["target_update_interval"],
            tensorboard_log=f"{exp_dir}/tb",  
            verbose=hp_cfg["model"]["verbose"],
            seed=hp_cfg["seed"],
            device=hp_cfg["device"]
        )
    elif algo_name == "DDPG":
        model = DDPG(
            policy=hp_cfg["model"]["policy"], 
            policy_kwargs=policy_kwargs, 
            env=env,
            learning_rate=float(hp_cfg["model"]["learning_rate"]),
            batch_size=hp_cfg["model"]["batch_size"],
            buffer_size=hp_cfg["model"]["buffer_size"],
            learning_starts=hp_cfg["model"]["learning_starts"],
            tau=hp_cfg["model"]["tau"],
            gamma=hp_cfg["model"]["gamma"],
            train_freq=hp_cfg["model"]["train_freq"],
            gradient_steps=hp_cfg["model"]["gradient_steps"],
            tensorboard_log=f"{exp_dir}/tb",  
            verbose=hp_cfg["model"]["verbose"],
            seed=hp_cfg["seed"],
            device=hp_cfg["device"]
        )
    elif algo_name == "TD3":
        model = TD3(
            policy=hp_cfg["model"]["policy"], 
            policy_kwargs=policy_kwargs, 
            env=env,
            learning_rate=float(hp_cfg["model"]["learning_rate"]),
            batch_size=hp_cfg["model"]["batch_size"],
            buffer_size=hp_cfg["model"]["buffer_size"],
            learning_starts=hp_cfg["model"]["learning_starts"],
            tau=hp_cfg["model"]["tau"],
            gamma=hp_cfg["model"]["gamma"],
            train_freq=hp_cfg["model"]["train_freq"],
            gradient_steps=hp_cfg["model"]["gradient_steps"],
            target_policy_noise=hp_cfg["model"]["target_policy_noise"],
            target_noise_clip=hp_cfg["model"]["target_noise_clip"],
            tensorboard_log=f"{exp_dir}/tb",  
            verbose=hp_cfg["model"]["verbose"],
            seed=hp_cfg["seed"],
            device=hp_cfg["device"]
        )
    # elif algo_name == "A2C":
    #     model = A2C(
    #         policy=hp_cfg["model"]["policy"], 
    #         policy_kwargs=dict(
    #             net_arch=[128, 128],
    #             optimizer_kwargs=dict(eps=1e-5)  # <-- explicitly set eps
    #         ),
    #         env=env,
    #         learning_rate=float(hp_cfg["model"]["learning_rate"]),
    #         n_steps=hp_cfg["model"]["n_steps"]//num_proc,
    #         gamma=hp_cfg["model"]["gamma"],
    #         gae_lambda=hp_cfg["model"]["gae_lambda"],
    #         ent_coef=hp_cfg["model"]["ent_coef"],
    #         vf_coef=hp_cfg["model"]["vf_coef"],
    #         max_grad_norm=hp_cfg["model"]["max_grad_norm"],
    #         rms_prop_eps=hp_cfg["model"]["rms_prop_eps"],
    #         use_rms_prop=hp_cfg["model"]["use_rms_prop"],
    #         tensorboard_log=f"{exp_dir}/tb",  
    #         verbose=hp_cfg["model"]["verbose"],
    #         seed=hp_cfg["seed"],
    #         device=hp_cfg["device"]
    #     )
    else:
        raise NotImplementedError(f"Algorithm {algo_name} not supported in this script yet.")
    

    # Train the PPO model
    model.learn(
        total_timesteps=hp_cfg["algo"]["total_timesteps"], 
        callback=WandbCallback(
            model_save_path=f"{exp_dir}/models",
            model_save_freq=1000,
            verbose=2
        ), 
        log_interval=hp_cfg["algo"]["log_interval"], 
        
    )

    
    return None



def get_env_cfg():
    '''
    loads environment param cfg from yaml file path and params provided by args
    '''
    with open(f"{ROOT}/{ARGS.C_path}", 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.SafeLoader)


    if env_cfg["env"]["net"] != 'SiouxFalls':
        raise NotImplementedError("Only SiouxFalls has been implemented.")

    # load the network
    if env_cfg['env']['net'] == 'SiouxFalls':
        net_dir = 'SiouxFalls_net.csv'
    G = net.load_network(net_dir)
    env_cfg['env']['T_T'] = net.T_T_matrix(G , np.array(env_cfg["env"]['speeds']), env_cfg["env"]['n_nodes'] , len(np.array(env_cfg["env"]['etas'])))

    
    return env_cfg

def get_hp_cfg():
    '''
    loads  hyperparam cfg from yaml file path and params provided by args
    '''
    with open(f"{ROOT}/{ARGS.HP_path}", 'r') as f:
        hp_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # next we update any meta-parameters that can be set before itearting over seeds
    hp_cfg['project_name'] = f"RL_{ARGS.project_id}"
    return hp_cfg


def run_seed(env_cfg, hp_cfg ,save_dir):
    '''
    runs one instance corresponding to this seed
    NOTE: will skip if this run already exists in our save_dir
    '''
    # create directory for this seed, making sure it does not exist
    exp_dir = f"{save_dir}/{hp_cfg['run_name']}"
    try: 
        os.mkdir(exp_dir)
        os.mkdir(f"{exp_dir}/models")
    except: print(f"WARNING: skipping duplicate run:\n\t{hp_cfg['run_name']}")

    config = dict(env_cfg)
    config.update(hp_cfg)
    # setup wandb
    wandb.init(
        # set the wandb project where this run will be logged
        #entity = "Zibaie_RL", # NOTE: this is your account, change accordingly
        project=hp_cfg['project_name'], # this is our project, defined by env name
        dir=exp_dir, # this is where everything is saved, for now we do not
        name=hp_cfg['run_name'], # this will be used to display our run on wandb (and save here)
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        config = config # this is passed on for tracking the experiment directly (and saved automatically by wandb)
    )
    train(env_cfg, hp_cfg, exp_dir)
    wandb.finish()
    # processing and manual saving can go here
    return None


def main():
    # first we load the env param used for this experiment
    env_cfg = get_env_cfg()

    # next we load the hyperparameters used for this experiment
    hp_cfg_base = get_hp_cfg()


    # create base directory recursively
    save_dir = f"{ROOT}/{ARGS.result_dir}/{env_cfg['env']['name']}/{hp_cfg_base['model']['name']}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to:\n\t{save_dir}")
    # setup seeding and run all instances iteratively (parallelization should be implemented with bash scripts directly)
    # extract config name for identifying run
    hp_name = ARGS.HP_path.split('/')[-1].split('.')[0]
    param_name = ARGS.C_path.split('/')[-1].split('.')[0]
    for seed in range(ARGS.start_seed, ARGS.start_seed+ARGS.n_seeds):
        # create a config specific to the seed we want to run
        hp_cfg = dict(hp_cfg_base)
        hp_cfg['seed'] = seed
        hp_cfg['run_name'] = f"{param_name}_{hp_name}_{ARGS.run_name}_{seed}" 
        # ideally, we do not return anything and process everything after!
        run_seed(env_cfg, hp_cfg , save_dir)   

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--project_id', type=str, default='FoodDElivery', help="additional identiffier used for project name")
    parser.add_argument('--run_name', type=str, default='run', help="all runs saved to the wandb project will use config info ,run_name and seed for identification")
    parser.add_argument('--start_seed', type=int, default=0, help="seed to start from")
    parser.add_argument('--n_seeds', type=int, default=1, help="number of seeds to run")
    parser.add_argument('--HP_path', type=str, default='hyperparameters/H1.yaml', help="location of yaml config file to use for hyperparameters, relative to ROOT dir of project")
    parser.add_argument('--C_path', type=str, default='env_config/env_c20.yaml', help="location of yaml config file to use for environment parameters, relative to ROOT dir of project")
    parser.add_argument('--result_dir', type=str, default='results', help="name of directory where results are stored. Curerntly using env->model to save the result.")
    
    ARGS = parser.parse_args()
    print(ARGS)
    main()