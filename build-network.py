# this module defines methods for getting adjacency matrix and degree array 
import numpy as np
import networkx as nx
import pandas as pd

def nx_er(n, ave_deg):
    ''' build ER network from networkx and return adjacency matrix and degree array.

    '''
    nw = nx.erdos_renyi_graph(n,ave_deg/(n-1))    # build network 
    #k = np.array(nw.degree().values())    # np array of degree for each node
    adjacency = nx.adjacency_matrix(nw)    # sparse adjacency matrix of nw
    adjacency = np.array(adjacency.todense())    # dense matrix of nw, np array
    return adjacency


def nx_sf(n, m):
    ''' build Barabasi-Albert SF network from networkx and return adjacency matrix and degree array.
       
        m is the smallest degree.
    '''
    nw = nx.barabasi_albert_graph(n,m)    # build network
    #k = np.array(nw.degree().values())    # np array of degree for each node
    adjacency = nx.adjacency_matrix(nw)    # sparse adjacency matrix of nw
    adjacency = np.array(adjacency.todense())    # dense matrix of nw, np array
    return adjacency

def stocks_nw(ts):
    ''' build correlation network from stock correlation data. 

        ts is the threshold that if the correlation is below ts, then there is no link, vise versa.
    '''
    path = r'/Users/seanshao/Documents/research/syn/ss/codes/'    #folder name
    df = pd.read_csv(path + 'cor_coef.csv', index_col = 0)    # read in correlation matrix 
    coef_matrix = np.array(df)    # convert to np array matrix
    adjacency = 1*(coef_matrix >= ts)    # convert to binary, above threshold 1, otherwise 0
    for i in range(len(adjacency)):
        adjacency[i][i] = 0     # node doesn't connect to themselves
    #k = adjacency.sum(axis=0)    # degree for each company
    return adjacency
