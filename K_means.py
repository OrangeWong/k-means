# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:49:34 2017

@author: richa
"""

import numpy as np
from scipy.spatial import distance

MAX_ITERATIONS = 50
TORR = 1e-5

def get_centroids_from_data(data, k):
    # to select k centroids from data
    np.random.seed(0)
    selected_rows = np.random.randint(0, get_num_observations(data), 
                                      size = k)
    return np.array(data)[selected_rows, :]
    
def get_num_observations(data):
    return (np.array(data)).shape[0]

def get_num_features(data):
    # return the number of features based on input data.
    data = np.array(data)
    return data.shape[1]

def get_labels(data, centroids):
    # to obtain the labels in the dataset based on the input centroids
    return distance.cdist(data, centroids).argmin(axis = 1)

def is_centroids_complete(labels, k):
    return (~np.isin(labels, range(k+1))).sum() == 0
    
def get_centroids(data, labels, k):
    while not is_centroids_complete(labels, k):
        centroids = get_centroids_from_data(data, k)
        labels = get_labels(data, centroids)
    list_centroids = [np.average(data[labels == j, ], axis =0) for j in range(k)]
    return np.vstack(list_centroids)

def should_stop(old_centroids, centroids, iters):
    if iters > MAX_ITERATIONS:
        return True
    else:
        if old_centroids is None:
            return False
        else:
            return (np.abs(old_centroids - centroids).sum() < TORR)
        
def kmeans(data, k):
    # initiating centroids, iteration, old_centroids 
    centroids = get_centroids_from_data(data, k)
    iters = 0
    old_centroids = None
    
    # running the k means
    while not should_stop(old_centroids, centroids, iters):
        # assigning
        old_centroids = centroids
        iters += 1
        print(old_centroids)
        # calculating the labels and centroids
        labels = get_labels(data, centroids)
        centroids = get_centroids(data, labels, k)
        
    # return centorids when the stop condition is true
    return centroids  