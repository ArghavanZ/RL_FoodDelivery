
from enum import Enum
from collections import deque
from typing import Callable
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from stable_baselines3.common.utils import set_random_seed
# all HELPER functions will be loaded as u.HELPER
from ..utils import util as u

## The action is the difference of the price for each courier modality and maximum price
## We have the maximum utility (maximum cost in terms of both latency and price) for each customer

def make_env(params: dict, 
             render_mode: str = None, 
             rank: int = 0, 
             seed: int = 0,
             timelimit: int = 100) -> Callable:
    """
    Utility function specifically for gridworld and stablebaselines3
    
    Inputs
        params:      requires paramter set imported from yaml or manually
        render_mode: type of rendering to use, NotImplemented
        rank:        added to seed to allow for Vector Env
        seed:        seed for random number generator 
        timelimit:   manually set
    """
    def _init() -> gym.Env:
        env = MMDelivery(params, render_mode)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=timelimit)
        env.reset(seed = seed + rank)
        return env

    set_random_seed(seed)
    return _init
    
class MMDelivery(gym.Env):

    """
    ### We did not implement render modes, so this is just boilerplate code
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, 
                 params,
                 render_mode = None, 
                ):
        """
        NOTE: Default params are not provided
        
        Current assumptions:
        - No probability of couriers leaving or entering
        - couriers are generated uniformly provided with num_couriers
        - termination after max_q orders are passed through the queue (pass or fail)
        - Action is directly setting prices for CLOSEST couriers
        """

        self.params = params
        # LOAD EACH VARIABLE ANYWAY FOR CONVENIENCE
        
        self.window_size = 512  # The size of the PyGame window
        # paramss that scale the problem

        #self.size = params["size"] # float, The size of the circle is its circumference 

        self.len_q = params["len_q"] # int, length of the queue that we OBSERVE 
        self.max_q = params["max_q"] # int, length of the entire queue, TERMINATION CONDITION 

        
        self.num_couriers = np.array(params["num_couriers"]) # list of int, length M, number of couriers of each modality
        self.modalities = np.array(params["modalities"]) # list of str, length M, represents courier modes
        self.courier_init = np.array(params["courier_init"]) # initial latency is initialized as uniform between these parameters

        # params defining the queue generation
        self.rate = params["rate"] # int, the total rate of the system (orders)
        self.rates_p = np.array(params["rates_p"]) # list of float, relative pickup rates (sums to 1, length N)
        self.rates_d = np.array(params["rates_d"]) # list of float, relative dropoff rate (sums to 1, length N)

        # maximum units/limits
        self.max_exp = params["max_exp"] # int (int), The waiting time limit for orders expiring
        self.max_lat = params["max_lat"] # float, The max latency users will wait
        self.max_tau = params["max_tau"] # float, The max price users will pay
        self.max_u = params["max_utility"] # float, The max utility users will pay

        # courier params
        self.vot_means = params["vot_means"] # float, mean vot of each region
        self.vot_stds = params["vot_stds"] # float, std vot of each region

        
        self.etas = np.array(params["etas"]) # list of float, length M, cost per unit time conversion for each courier 
        self.T_T = np.array(params["T_T"]) # np matrix of floats, where T_T[i,j,k] is the time to travel from region i to j for courier k

        # DERIVED params and checks 
        self.N = params["n_nodes"]  # The number of regions (nodes!)
        self.L = params["n_links"] # The number of links (edges) in the network
        self.M = len(self.modalities) # The number of courier modalities

        #self.regions = np.arange(self.N)# list of int, length N, regions (nodes) in the network
        self.regions = np.arange(1, self.N+1) # list of int, length N, regions (nodes) in the network
                
        # TODO: Render modes are not implemented, left boiler plate code inplace
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Flattened observation space
        self.observation_space = spaces.Dict(
            {
                "o_p": spaces.Box(
                    low=0,
                    #high=self.N,
                    high=self.N+1 ,
                    shape=(self.len_q,),
                    dtype=np.int32
                ),
                "o_d": spaces.Box(
                    low=0,
                    #high=self.N,
                    high=self.N+1, 
                    shape=(self.len_q,),
                    dtype=np.int32
                ),
                "o_l": spaces.Box(
                    low=0,
                    high=self.max_exp,
                    shape=(self.len_q,),
                    dtype=np.int32
                ),
                "c_j": spaces.Box(
                    low=0,
                    high=self.M,
                    shape=(sum(self.num_couriers),),
                    dtype=np.int32
                ),
                "c_d": spaces.Box(
                    #low=0,
                    low=1,
                    #high=self.N,
                    high=self.N+1, 
                    shape=(sum(self.num_couriers),),
                    dtype=np.int32
                ),
                "c_l": spaces.Box(
                    low=0,
                    high=self.max_u*10,
                    shape=(sum(self.num_couriers),),
                    dtype=np.float32
                ),
                "Latencies": spaces.Box(
                    low=0,
                    high= self.max_lat*5,
                    shape=(self.M,),
                    dtype=np.float32
                ),
            }
        )

        # directly set prices for closest couriers ("Latencies")
        self.action_space = spaces.Box(
            low=0,
            high=self.max_tau,
            shape=(self.M,),
            dtype=np.float32
        )
        

    def _get_obs(self):
        
        
        o_p = np.zeros(self.len_q, dtype=np.int32)
        o_d = np.zeros(self.len_q, dtype=np.int32)
        o_l = np.zeros(self.len_q, dtype=np.int32)

        if len(self.queue) > 0:
            np_queue = np.array(self.queue) 
            num_orders = int(min(len(self.queue), self.len_q)) # Ensure we do not exceed the length of the observed queue
        # Initialize arrays with zeros (or any default value)
        # Fill in the actual data
            o_p[:num_orders] = np_queue[:num_orders, 0].astype(np.int32)
            o_d[:num_orders] = np_queue[:num_orders, 1].astype(np.int32)
            o_l[:num_orders] = np_queue[:num_orders, 2].astype(np.int32)
        # Similarly, ensure other components have consistent shapes
        
        observation = {
            "o_p": o_p,
            "o_d": o_d,
            "o_l": o_l,
            "c_j": self.couriers[:, 0].astype(np.int32),
            "c_d": self.couriers[:, 1].astype(np.int32),
            "c_l": self.couriers[:, 2].astype(np.float32),
            "Latencies": self.star_lat.astype(np.float32),
        }
        return observation

    def _get_info(self):
        '''
        Get additional information about the environment.

        Returns:
        - info: Dictionary of additional information
        '''
        if self.chosen_ord:
            info = {
                "time_step": self.t,
                "ord": 1,
                "tau": self.chosen_price,
                "lat": self.chosen_latency,
                "utility": self.chosen_utility,
                "mode": self.chosen_mode,
                "region":self.chosen_reg,
                "Queue_Length": len(self.queue),
                "arrivals": self.idx_order,
            }
        else:
            info = {
                "time_step": self.t,
                "ord": 0,
                "tau": 0,
                "lat": 0,
                "utility": 0,
                "mode": None,
                "region":None,
                "Queue_Length": len(self.queue),
                "arrivals": self.idx_order,
            }

        return info

        
    def reset(self, seed=None, options=None):
        """
        Initialize the environment using current parameters
        
        NOTE: options not used currently
        """
        super().reset(seed=seed)
        # seed action space
        self.action_space.seed(seed)
        
        ### CODE GOES HERE ###
        self.t = 0  # Current time step
        self.idx_order = 0  # Index for upcoming orders
        # generate all orders
        self.orders = u.gen_orders(self.np_random, self.rates_p, self.rates_d, self.regions, self.max_q)
        # initialize the queue 
        self.queue = deque()
        # initialize the couriers
        self.couriers = u.gen_couriers(self.np_random, self.num_couriers, self.courier_init, self.regions)
        # get all latencies (no orders in the queue yet, thelatency is just the courier's latency)
        self.all_lat = self.couriers[:, 2].copy()  # Copy the latencies of the couriers
        # find the M closest couriers (one for ea modality), and their indeces
        self.star_lat, self.star_idx = u.get_star_lat(self.all_lat, self.couriers[:,0], self.M)  
        observation = self._get_obs()

        # these are needed for info
        self.chosen_latency = 0
        self.chosen_price = 0
        self.chosen_utility = 0
        self.chosen_mode = None
        self.chosen_reg = None
        self.chosen_ord = False
        self.num_arrivals = 0

        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        # (1) Increment time and orders so that new orders added do not get affected
        #----------------------------------------------------------------------------
        self.t += 1
        # Update waiting time o_l for all orders in the queue
        for order in self.queue:
            order[2] += 1  # Increase o_l by 1
        # Remove orders that have waited longer than max_exp
        expired_orders = [o for o in self.queue if o[2] > self.max_exp]
        for o in expired_orders:
            # Remove expired order
            self.queue.remove(o)
        
        # Initialize termination condition, will trigger if we run out of orders
        terminated = False
        # Initialize rewards
        reward = 0

        # Get the current order from the queue
        # If there are orders in the queue, set the current order to the first one
        if len(self.queue) > 0:
            current_order = self.queue[0]


        else:
            # No orders in the queue, reset current order
            current_order = np.zeros(3)
            #### if no orders in the queue, we go to the next step with reward 0

            ### update for next step
            self.num_arrivals = int(self.np_random.poisson(self.rate))  # Poisson arrival process
            # Append a new order if available and total_processed_orders < max_q
            if self.num_arrivals > 0:
                for _ in range(self.num_arrivals):
                    if self.idx_order < len(self.orders):
                        self.queue.append(self.orders[self.idx_order].tolist())
                        self.idx_order += 1
                # if not available, we are done
                    else:
                        terminated = True
            # Decrease c_l for all couriers by 1 time unit (simulate time passing)
            self.couriers[:, 2] = np.maximum(0, self.couriers[:, 2] - 1 )
                    
            self.chosen_ord = False
            self.chosen_latency = 0
            self.chosen_price = 0
            self.chosen_utility = 0
            self.chosen_mode = None
            self.chosen_reg = None
            info = self._get_info()
            observation = self._get_obs()
            return observation, 0, terminated, False, info


        ### Check if a new order is coming in 
        self.num_arrivals = int(self.np_random.poisson(self.rate))  # Poisson arrival process
            # Append a new order if available and total_processed_orders < max_q
        if self.num_arrivals > 0:
            for _ in range(self.num_arrivals):
                if self.idx_order < len(self.orders):
                    self.queue.append(self.orders[self.idx_order].tolist())
                    self.idx_order += 1
                # if not available, we are done
                else:
                    terminated = True

        # Decrease c_l for all couriers by 1 time unit (simulate time passing)
        self.couriers[:, 2] = np.maximum(0, self.couriers[:, 2] - 1 )
        
        
        # (2) Compute transition after taking action
        #### if no current order, Just return 0 reward
        
        #----------------------------------------------------------------------------
        # NOTE: action defines how much less than max price we charge!
        # Ensure action is within valid bounds
        # debugging
        
        max_action = self.action_space.high[0]
        action = (np.ones(action.shape)*max_action) - action
        # end debugging
        tau = np.clip(action, 0, self.max_tau)  # Prices τ_j(t)
        # tau = np.array(self.max_tau)
        # end note

        
        
        # Get dropoff region index for current order
        o_d = int(current_order[1])
        
        # Sample user's Value of Time (VoT)
        alpha_mean = self.vot_means[o_d-1] # Indexing from 0, so o_d-1
        alpha_std = self.vot_stds[o_d-1] # Indexing from 0, so o_d-1
        alpha_t = self.np_random.normal(alpha_mean, alpha_std)
        alpha_t = max(alpha_t, 0.001)  # Ensure VoT is positive

        # Compute utilities for each modality using stored latencies
        utilities = alpha_t * self.star_lat + tau  # u_j(t) = α(t)·ℓ_j(t) + τ_j(t)
        # Apply constraints
        feasible = (utilities <= self.max_u)

        if np.any(feasible):
            # Feasible option exists
            feasible_indices = np.where(feasible)[0]

            # Select the modality with minimal utility
            chosen_modality = feasible_indices[np.argmin(utilities[feasible_indices])]

            # Collect all data you need
            self.chosen_ord = True
            self.chosen_latency = self.star_lat[chosen_modality]
            self.chosen_price = tau[chosen_modality]
            self.chosen_utility = utilities[chosen_modality]
            self.chosen_mode = chosen_modality
            self.chosen_reg = int(o_d)
            # rest of data
            eta_j = self.etas[chosen_modality]
            courier_idx = self.star_idx[chosen_modality]  # Get courier index
            # Update the chosen courier's c_l and c_d
            self.effective_lat = self.chosen_latency - self.couriers[courier_idx, 2]
            self.couriers[courier_idx, 2] = self.chosen_latency  # Update c_l
            self.couriers[courier_idx, 1] = current_order[1]  # Update c_d to o_d

            # Remove the order from the queue
            self.queue.popleft()
                
            # NOTE: Compute reward
            reward = self.chosen_price - eta_j * self.effective_lat
            # reward = -1 * self.effective_lat
            # reward = -1 * self.chosen_latency
        else:
            # No feasible options; order remains in the queue
            # No reward in this case
            # explicitly set to zero so get_info is default
            self.chosen_ord = False
            pass

        # (3) Recopute relevant parameters for next timestep (messy ending condition, but for completeness)
        #----------------------------------------------------------------------------            
        if len(self.queue) > 0:
            current_order = self.queue[0]
            self.all_lat = u.get_all_lat(current_order, self.couriers, self.T_T)
        else: 
            # No orders in the queue, reset current order
            current_order = np.zeros(3)
            self.all_lat = self.couriers[:, 2].copy()

        if self.idx_order >= len(self.orders):
            # No more orders in the queue, reset the queue
            terminated = True
            self.queue = deque()
            current_order = np.zeros(3)
            self.all_lat = self.couriers[:, 2].copy()
            
        
        self.star_lat, self.star_idx = u.get_star_lat(self.all_lat, self.couriers[:, 0], self.M)

        # Prepare observation and info
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()


        return observation, reward, terminated, False, info




    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        raise NotImplementedError("TODO: No render modes implemented, set render_mode to None")

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
