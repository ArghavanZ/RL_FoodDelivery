
import os, sys
ROOT = os.path.abspath(os.curdir)
sys.path.append(os.path.abspath(os.path.join(ROOT,'src')))

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt




# Load the network from a CSV file given the network name 
def load_network(file_path = 'SiouxFalls_net.csv'):
    '''
    Load the network from a CSV file.
    The CSV file should have columns 'A', 'B', and 'a0' representing the edges and their weights.
    The default file path is 'SiouxFalls_net.csv'.
    Returns a NetworkX DiGraph object representing the network.
    '''
    file_path = f"RL_FoodDelivery/network/data/{file_path}"
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    G = nx.from_pandas_edgelist(df, 'A', 'B', edge_attr='a0', create_using=nx.DiGraph())
    return G
    
def load_pos(file_path = 'SiouxFalls_node.csv'):
    '''
    Load the nodes position from a CSV file.
    The CSV file should have columns 'Node' , 'X' and 'Y'  representing the nodes and their positions.
    The default file path is 'SiouxFalls_node.csv'.
    Returns a numpy array of the nodes positions and a dictionary of the nodes positions.
    '''
    file_path = f"RL_FoodDelivery/network/data/{file_path}"
    
    # Read the CSV file into a DataFrame
    pos = pd.read_csv(file_path , index_col='Node')
    pos_n = pos.copy()
    pos_n['X'] = pos_n['X']/100000
    pos_n['Y'] = pos_n['Y']/100000
    pos = pos.to_dict('index')
    for i in pos:
        x = pos[i]['X']/10000
        y = pos[i]['Y']/10000
        pos[i] = (x, y)

    pos_np = pos_n.to_numpy()
    return pos_np , pos

# Calculate T_T matrix 
def T_T_matrix(G, speed , N , M):
    
    T_T = np.zeros((N, N, M))
    # Change the a0 attribute of the edges based on the speed of the courier
    
    for j in range(M):
        G_p = G.copy()
        i = 0
        for u, v, data in sorted(G_p.edges(data=True)):
            data['a0'] = data['a0'] / speed[i][j] ### is this change the weight in the graph or not? 
            i+=1
        paths = dict(nx.all_pairs_dijkstra_path_length(G_p, weight="a0"))   
        for k in range(N):
            for l in range(N):
                T_T[k][l][j] = paths[k+1][l+1]  # Get the shortest path length or inf if no path exists
        
    return T_T.tolist()


def T_T_C_D(G, pos_d, d_s  , N , M):
    '''
    Calculate the T_T matrix for car network and drone network with different network shape. 
    For drone, the network is a complete graph.
    '''

    T_T = np.zeros((N, N, M))
    # There is no speed for drones or cars for now. But since we want to have a control over latencies, we can add speed for drones and cars later with uncommenting the following lines. 
    # Also changing the inputs of the function to include speed for drones and cars.
    # Change the a0 attribute of the edges based on the speed of the courier    
    # 
    G_p = G.copy()
    # i = 0
    # for u, v, data in sorted(G_p.edges(data=True)):
    #     data['a0'] = data['a0'] / speed_c[i] 
    #     i+=1

    #### Car network
    paths = dict(nx.all_pairs_dijkstra_path_length(G_p, weight="a0"))
    for k in range(N):
        for l in range(N):
            T_T[k][l][0] = paths[k+1][l+1]  # Get the shortest path length or inf if no path exists
    
    #### Drone network
    for k in range(N):
        for l in range(N):
            pk = pos_d[k]
            pl = pos_d[l]
            dist = (pk - pl)/d_s
            # dist = dist / speed_d[k][l]
            T_T[k][l][1] = np.linalg.norm(dist)  # Get the shortest path length or inf if no path exists
    
    return T_T

    
    
