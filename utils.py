import numpy as np
import networkx as nx

def gaussian_normalization(train_x):
    mu = np.mean(train_x, axis=0)
    dev = np.std(train_x, axis=0)
    norm_x = (train_x - mu) / (dev + 1e-12)
    # print norm_x
    return norm_x

def min_max_normalization(train_x):
    _max = np.max(train_x, axis=0)
    _min = np.min(train_x, axis=0)
    norm_x = (train_x - _min) / (_max-_min)
    # print norm_x
    return norm_x

def zero_max_normalization(train_x):
    _max = np.max(train_x, axis=0)
    norm_x = train_x / _max
    # print norm_x
    return norm_x



def get_edges(G,self_edge=False):
    edges = set()
    u_i = []
    u_j = []
    label = []
    for edge in G.edges():
        if (edge[0],edge[1]) in edges:
            continue
        u_i.append(edge[0])
        u_j.append(edge[1])
        dat = 1
        label.append(dat)        
        edges.add((edge[0],edge[1]))
    if self_edge:
        for node_index in range(nx.number_of_nodes(G)):
            if (node_index,node_index) in edges:
                continue
            u_i.append(node_index)
            u_j.append(node_index)
            label.append(1)
    return u_i, u_j, label

