## File to store extra functions required by the environemnt

import numpy as np

def gen_orders(np_random, rates_p, rates_d, regions, o):
    '''
    Generate o orders with pickup and dropoff locations based on provided rates and regions.
    Returns a numpy array of shape (o, 3) with columns [o_p, o_d, o_l].

    Parameters:
    - np_random: passed numpy generator for consistency
    - rates_p: Array of pickup rates for each region (normalized to sum to 1)
    - rates_d: Array of dropoff rates for each region (normalized to sum to 1)
    - regions: Array of region indices (e.g., [1, 2, ...])
    - o: Number of orders to generate
    '''
    
    orders = np.zeros((o, 3))  # Each order has [o_p, o_d, o_l]

    # Normalize rates to sum to 1
    rates_p = rates_p / np.sum(rates_p)
    rates_d = rates_d / np.sum(rates_d)

    # Generate pickup and dropoff region indices
    pickup_regions = np_random.choice(regions, size=o, p=rates_p)
    dropoff_regions = np_random.choice(regions, size=o, p=rates_d)

    # Generate pickup and dropoff locations within regions
    for i in range(o):
        # Pickup location
        o_p = pickup_regions[i]

        # Dropoff location
        o_d = dropoff_regions[i]

        # Order waiting time initialized to 0
        o_l = 0

        orders[i] = [o_p, o_d, o_l]

    return orders

    
def gen_couriers(np_random, num_couriers, courier_init, regions):
    '''
    Generate couriers for each modality, uniformly assigned tp drop off locations.
    Returns a numpy array of shape (total_couriers, 3) with columns [c_j, c_d, c_l].

    Parameters:
    - num_couriers: Array with the number of couriers for each modality (length M)
    - courier_init: Tuple (min_time, max_time)
    - regions: Array of region indices (e.g., [1, 2, ...])
    - np_random: passed numpy generator for consistency
    
    '''
    M = len(num_couriers)
    total_couriers = np.sum(num_couriers)
    couriers = np.zeros((total_couriers, 3))  # Each courier has [c_j, c_d, c_l]

    idx = 0
    for j in range(M):
        n_j = num_couriers[j]
        for _ in range(n_j):
            c_j = j  # Modality index
            c_d = np_random.choice(regions)  # Next dropoff Location
            c_l = np.max((0,np_random.uniform(courier_init[0],courier_init[1])))  # Time remaining until available
            couriers[idx] = [c_j, c_d, c_l]
            idx += 1

    return couriers


#### we do not use this function, but use it as the inline function in the get_all_lat function.
def compute_travel_time(x, x_prime, c_j, T_T , N):
    '''
    Compute minimal travel time between two nodes on the circle for a given modality,
    accounting for varying speeds in different regions.

    NOTE: Since we do not consider congestion over time, we assume that the travel time is constant between two
    nodes, the origin and the destination. 
    This allow us to compute the shortest path at the start of the first episode, and then use it for the rest of the episode.
    This is a simplification of the real world, but it allows us to focus on the learning of the policy. 
    Note that the congestion difference between the regions is considered in this implementation since the travel time is different different links.

    NOTE: We have a big matrix of distances between all the nodes,
    but since the path between nodes are unidirectional, this matrix is symmetric.
    
    Parameters:
    - x: Origin location (int)
    - x_prime: Destination location (int)
    - c_j: Modality index (int)
    - T_T: 3D array of travel times [regions x regions x modality]
    - N: Number of regions (int)

    Returns:
    - Minimal travel time between x and x_prime for modality c_j (float)

    '''
    # Compute the shortest path between x and x_prime
    # We use the distance matrix to compute the shortest path
    
    if x == x_prime:
        t = 0 # If the origin and destination are the same, the travel time is negligible. 
    else:
        t = T_T[x][ x_prime][c_j] # Get the travel time between x and x_prime for modality c_j
    return t
    

def get_all_lat(order, couriers, T_T):
    '''
    Compute latencies for all couriers for the given order.

    Parameters:
    - order: Array [o_p, o_d, o_l]
    - couriers: Array of couriers [c_j, c_d, c_l]
    - regions: List of regions
    - T_T: 3D array of travel times [regions x regions x modality]

    Returns:
    - all_lat: Array of latencies for each courier
    '''
    
    o_p = int(order[0])-1 # Order's pickup location (indexing from 0)
    o_d = int(order[1])-1 # Order's dropoff location (indexing from 0)
    num_couriers = couriers.shape[0]
    all_lat = np.zeros(num_couriers)

    for idx in range(num_couriers):
        c = couriers[idx]
        c_j = int(c[0])
        
        c_d = int(c[1]) - 1 # Courier's dropoff location (indexing from 0)
        c_l = c[2]

        # Compute travel times
        t_cd_op = t = T_T[c_d][ o_p][c_j]# Travel time from courier's dropoff to order's pickup
        #compute_travel_time(c_d, o_p, c_j, T_T , N) 
        t_op_od = T_T[o_p][o_d][c_j]
        #compute_travel_time(o_p, o_d, c_j, T_T , N) # Travel time from order's pickup to order's dropoff

        # Total latency
        latency = c_l + t_cd_op + t_op_od
        all_lat[idx] = latency

    return all_lat



def get_star_lat(all_lat, c_js, M):
    '''
    Find the minimal latency for each modality.

    Parameters:
    - all_lat: Array of latencies for all couriers
    - c_js: Array of modality indices for all couriers
    - M: Number of modalities

    Returns:
    - star_lat: Array of minimal latencies for each modality (length M)
    - star_idx: Array of corresponding courier indices (length M)
    '''
    star_lat = np.zeros(M, dtype=np.float32)
    star_idx = np.zeros(M, dtype=np.int32)

    for j in range(M):
        # Indices of couriers of modality j
        indices_j = np.where(c_js == j)[0]
        if len(indices_j) > 0:
            latencies_j = all_lat[indices_j]
            min_idx = np.argmin(latencies_j)
            star_lat[j] = latencies_j[min_idx]
            star_idx[j] = indices_j[min_idx]
        else:
            raise NotImplementedError("Must have at least one courier for each modality to provide a valid choice")

    return star_lat, star_idx