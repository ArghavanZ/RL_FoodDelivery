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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

# import internal packages
from RL_FoodDelivery.env.drone_delivery import MMDelivery, make_env
from  RL_FoodDelivery.utils import net
#----------------------#
#----------------------#

### Scenario2:


def run_n_evals(env, policy, policy_name,n, seed,  directory , env_cfg):

    wandb.init(
        # set the wandb project where this run will be logged
        #entity = "mbeliaev", # NOTE: this is your account, change accordingly
        project='RL_eval_final', # this is our project, defined by env name
        dir = directory, # this is where everything is saved, for now we do not
        name = f'{ARGS.run_name}_{policy_name}', # this will be used to display our run on wandb (and save here)
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        config= env_cfg
    )
    """
    Takes in env and policy, evaluates over n seeds 
    Inputs
        env - gym env (non vectorized for now)
        policy - function, policy(obs,input)-->action
        n - number of times to evaluate policy
        seed - initial seed to start evaluating from
    """
    N = env.unwrapped.N
    M = env.unwrapped.M
    all_stats = np.zeros(shape=(n,6+M+M),dtype=np.float32)
    reg_stats = np.zeros(shape=(n,N,4+M),dtype=np.float32)
    # computation 
    #-------------
    t_start = 0
    t_rew = 0
    for i_ep in range(n):
        # reset
        # mutual parameters, episode and region stats
        ep_uti, reg_uti = 0, np.zeros(N)
        ep_tau, reg_tau = 0, np.zeros(N)
        ep_lat, reg_lat = 0, np.zeros(N)
        ep_ord, reg_ord = 0, np.zeros(N)
        ep_modes, reg_modes = np.zeros(M), np.zeros((N,M)) 
        # parameters for epsiodes (all regions)
        ep_rew = 0
        ep_modes_uti = np.zeros(M)
        
        obs, info = env.reset(seed=seed+i_ep)
        # NOTE: As of now, env always returns truncated False, so it must be the timelimit wrapper if trigerred.
        terminated, truncated = False, False
        t = 0
        # run episode
        while not(terminated or truncated):
            t+=1
            t_start += 1
            action = policy(obs, info, env)
            obs, rew, terminated, truncated, info = env.step(action)
            t_rew += rew
            # wandb.log({"eval/step_reward": rew, 
            # "eval/cum_rew": t_rew, 
            # "eval/step": t_start,})
            ep_rew += rew
            i_reg = info['region']
            i_mode = info['mode']
            if i_reg is not None:
                # acquire rolling stats for ALL
                ep_uti += info['utility']
                ep_tau += info['tau']
                ep_lat += info['lat']
                ep_ord += info['ord']
                ep_modes[i_mode] += 1
                # acquire rolling stats for REG
                reg_uti[i_reg] += info['utility']
                reg_tau[i_reg] += info['tau']
                reg_lat[i_reg] += info['lat']
                reg_ord[i_reg] += info['ord']
                
                reg_modes[i_reg, i_mode] += 1
            # compute util of couriers
            for j_mode in range(M):
                idx_flag = obs['c_j'] == j_mode
                ep_modes_uti[j_mode] += sum(obs['c_l'][idx_flag] > 0)/sum(idx_flag)
        # check that episode ended due to truncation
        assert terminated == False, "Error: episode ended due to order queue running out"
        # push ep stats to dataset
        ep_modes_uti *= 100/t # average the utilization
        if ep_ord>0:
            all_stats[i_ep] = np.array([ep_modes[i] for i in range(M)]+
                                    [ep_modes_uti[i] for i in range(M)]+
                                    [ep_ord, ep_lat/ep_ord, ep_tau/ep_ord, ep_uti/ep_ord, ep_rew/t, ep_rew])
        
        wandb.log({"eval/episode_reward": ep_rew, 
        "eval/episode": i_ep,
        "eval/episode_length": t,
        "eval/episode_utility": ep_uti,
        "eval/episode_tau": ep_tau,
        "eval/episode_lat": ep_lat,
        "eval/episode_order": ep_ord})

        
        
        # now do this for regional stats
        for idx_reg in range(N):
            reg_stats[i_ep,idx_reg] = np.array([reg_modes[idx_reg,i] for i in range(M)]+
                                               [reg_ord[idx_reg], reg_lat[idx_reg]/reg_ord[idx_reg], reg_tau[idx_reg]/reg_ord[idx_reg], reg_uti[idx_reg]/reg_ord[idx_reg],])

    wandb.finish()        
    # processing
    #-------------
    stats_mean, stats_std = all_stats.mean(axis=0),all_stats.std(axis=0)
    reg_stats_mean, reg_stats_std = reg_stats.mean(axis=0),reg_stats.std(axis=0)
    
    return stats_mean, stats_std, reg_stats_mean, reg_stats_std

def policy_rand(obs, info, env):
    # seed depends on action_sapce.seed() 
    action = env.action_space.sample()
    return action

def policy_max(obs, info, env):
    action = np.zeros(obs["Latencies"].shape)
    return action

def policy_poor(obs, info, env):
    l_car, l_drone = obs["Latencies"] # assume shape of problem
    action = np.zeros(obs["Latencies"].shape)
    # normally we return max, max as before
    re = int(obs["o_d"][0])-1
    if re < 0 :
       
        # if no order, we do not change the action
        return action
    

    else:
        is_poor = any(x == re for x in [0,1,2,5,6,11,12,19,20,23])
        # if serving poor and car is slower than drone
        if is_poor and (l_car>l_drone):
            alpha = env.unwrapped.vot_means[re]
            # then we lower price of car just barely enough
            action[0] += alpha*(l_car-l_drone)
            # small correction
            action[0] += 0.01
        return action


def policy_best(obs, info, env):
    l_car, l_drone = obs["Latencies"] # assume shape of problem
    u_max = env.unwrapped.max_u # The maximum feasible utility
    tau_max = env.unwrapped.max_tau # The maximum feasible price
    dif = u_max - tau_max # The maximum possible latency considering VOT (VOT*latency)
    action = np.zeros(obs["Latencies"].shape) - dif #The base action (how lower we go from max price)
    
    # find the current order region
    re = int(obs["o_d"][0])

    if re == 0:
        # if no order, we do not change the action
        return action
    
    else:
    # mean VOT for the region
        alpha = env.unwrapped.vot_means[re-1]
        # cost per unit of couriers
        eta = env.unwrapped.etas

        # then we lower price a bit more than enough so that both couriers are feasible (almost same utility but a bit smaller for car since 0.01*l_c > 0.01*l_d)
        action += (alpha+0.01)*np.array([l_car,l_drone])
        # small correction (maybe not?)
        if (l_car>l_drone):
            action[1] += 0.02
        # check if we are having a positive reward
        if np.any(action+eta*np.array([l_car,l_drone]) > tau_max*np.ones(obs["Latencies"].shape)):
            action = tau_max*np.ones(obs["Latencies"].shape) - eta*np.array([l_car,l_drone])
        return action


def policy_RL(obs, info, env):
    action, _ = RL_MODEL.predict(obs)
    return action
#----------------------#
#----------------------#

def collect_evals(env_params , directory, env_cfg):
    # run data collection
    seed = 0
    n_ep = 100
    max_ep_len = 200
    env = Monitor(gym.wrappers.TimeLimit(MMDelivery(params=env_params), max_episode_steps=max_ep_len))
    # NOTE: assume that run_n_evals returns stats in direct correspondence to these columns 
    # other than policy
    stats_cols = ["policy",] + ["car", "drone"] + ["ca %", "drone %"] + ["ord", "lat", "tau", "uti" ,"rew/step", "rew" ]  # Add 'Policy' as the first column
    reg_stats_cols = ["region",] + ["car", "drone"] + ["ord", "lat", "tau", "uti",]  # Add 'Policy' as the first column
    policy_names = ["Random", "Max Price", "Zone Based","Max Order","RL"]
    policy_map = {"Random": policy_rand,
                "Max Price": policy_max,
                "Zone Based":policy_poor,
                "Max Order":policy_best,
                "RL":policy_RL,}

    all_data = {}
    for policy_name in policy_names:
        policy = policy_map[policy_name]
        all_data[policy_name] = run_n_evals(env, policy,  policy_name, n_ep, seed , directory, env_cfg) 

    return all_data, stats_cols, reg_stats_cols
    
# def get_base_cfg():
#     '''
#     loads base cfg from yaml file path and params provided by args
#     '''
#     with open(f"{ROOT}/{ARGS.config_path}", 'r') as f:
#         base_cfg = yaml.load(f, Loader=yaml.SafeLoader)
#     # NOTE: we do not update any meta params
#     return base_cfg



def get_env_cfg():
    '''
    loads environment param cfg from yaml file path and params provided by args
    '''
    with open(f"{ROOT}/{ARGS.C_path}", 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.SafeLoader)


    if env_cfg["env"]["net"] != 'SiouxFalls':
        raise NotImplementedError("Only SiouxFalls has been implemented for Gridworld environment.")

    # load the network
    if env_cfg['env']['net'] == 'SiouxFalls':
        net_dir = 'SiouxFalls_net.csv'
    G = net.load_network(net_dir)
    env_cfg['env']['T_T'] = net.T_T_matrix(G , np.array(env_cfg["env"]['speeds']), env_cfg["env"]['n_nodes'] , len(np.array(env_cfg["env"]['etas'])))
    
    return env_cfg

def main():
    # create base directory recursively
    print(f"Results will be saved to:\n\t{ARGS.save_path}")

    directory = ARGS.save_path.rsplit('/' , 1)[0]
    os.makedirs(directory, exist_ok=True)

    # If the file exists, ask whether to overwrite or append
    if os.path.exists(ARGS.save_path):
        response = input(
            f"File '{ARGS.save_path}' already exists. Overwrite (o) or append (a)? [o/a] "
        ).strip().lower()
        if response == 'o':
            # Overwrite means clear the file first
            open(ARGS.save_path, 'w').close()  # empty the file
            print(f"Overwriting '{ARGS.save_path}'.")
        elif response == 'a':
            print(f"Appending to '{ARGS.save_path}'.")
        else:
            print("Invalid input; defaulting to append.")
    else:
        print(f"File '{ARGS.save_path}' not found. It will be created.")

    

    # first we load the env param used for this experiment
    env_cfg = get_env_cfg()

    env_params = env_cfg["env"].copy()
    # Remove unused keys (base params)
    for key in ["name"]:
        env_params.pop(key, None)
    # Convert speeds and etas to numpy arrays and divide by 60
    env_params["etas"] = np.array(env_params["etas"]) 

    # evaluate
    all_data, stats_cols, reg_stats_cols = collect_evals(env_params, directory , env_cfg)
    col_width = 15

    env_write = env_cfg["env"].copy()
    for key in ["T_T", "speeds", "name"]:
        env_write.pop(key, None)
    # Open in append mode so we can add new entries
    with open(ARGS.save_path, 'a') as f:
        # first log config 
        line = "Writing results for config: "
        f.write(line)
        f.write("\n")
        for key, value in env_write.items():
            # Convert to string as needed
            if key == "T_T":
                pass

            f.write(f"{key}: {value}")
            f.write("\n")
        #--------------------#
        # output
        #--------------------#
        # Print results together first
        f.write("Evaluating Different Policies")
        f.write("\n")
        f.write("-" * (col_width * len(stats_cols)))
        f.write("\n")

        # Print column headers
        header = "".join(f"{col:<{col_width}}" for col in stats_cols)
        f.write(header)
        f.write("\n")
        f.write("-" * (col_width * len(stats_cols)))
        f.write("\n")
        # list results for all evaluations
        for policy_name, data in all_data.items():
            stats_mean, stats_std, _, _ = data
            row = f"{policy_name:<{col_width}}" + "".join(f"{mean:.1f} ± {std:.1f}".ljust(col_width) for mean, std in zip(stats_mean, stats_std))
            f.write(row)
            f.write("\n")

        # Now list indiv results
        f.write("\n")
        f.write("\n")
        f.write("Evaluating Policies Individually")
        f.write("\n")
        f.write("-" * (col_width * len(reg_stats_cols)))
        f.write("\n")
        for policy_name, data in all_data.items():
            f.write(f"Policy: {policy_name}")
            f.write("\n")
            f.write("-" * (col_width * len(reg_stats_cols)))
            f.write("\n")
            # Print column headers
            header = "".join(f"{col:<{col_width}}" for col in reg_stats_cols)
            f.write(header)
            f.write("\n")
            f.write("-" * (col_width * len(reg_stats_cols)))
            f.write("\n")
            # list results for all evaluations
            _, _, stats_mean, stats_std = data
            for i_reg in range(stats_mean.shape[0]):
                row = f"{i_reg+1:<{col_width}}" + "".join(f"{mean:.1f} ± {std:.1f}".ljust(col_width) for mean, std in zip(stats_mean[i_reg], stats_std[i_reg]))
                f.write(row)
                f.write("\n")

        print(f"\nMetrics have been written to '{ARGS.save_path}'.")


    # setup seeding and run all instances iteratively (parallelization should be implemented with bash scripts directly)
    # for seed in range(ARGS.start_seed, ARGS.start_seed+ARGS.n_seeds):
    #     # create a config specific to the seed we want to run
    #     exp_cfg = dict(base_cfg)
    #     exp_cfg['seed'] = seed
    #     exp_cfg['run_name'] = f"{ARGS.run_name}_{seed}" 
    #     # ideally, we do not return anything and process everything after!
    #     run_seed(exp_cfg, save_dir)   

    return None

if __name__ == "__main__":
   parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_path', type=str, default='eval/eval_env_c20_H1_r0/eval.txt', help="location where evaluation will be saved")
    parser.add_argument('--model_dir', type=str, default='results/drone_2/PPO/env_c20_H1_run_0/models/model.zip', help="location of model to evaluate")
    parser.add_argument('--C_path', type=str, default='env_config/env_c20.yaml', help="location of yaml config file to use for environment parameters, relative to ROOT dir of project")
    parser.add_argument('--run_name', type=str, default='eval_env_c20_H1_r0', help="all runs saved to the wandb project will use run_name for identification")
    
    ARGS = parser.parse_args()
    print(ARGS)
    RL_MODEL = PPO.load(ARGS.model_dir)
    main()