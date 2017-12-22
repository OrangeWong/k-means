# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:49:34 2017

@author: richa
"""

import numpy as np
from scipy.spatial import distance

class K_means():
    '''
    
    '''
    def __init__(self, n_clusters = 2, max_iter = 50, 
                 tol= 1e-4, random_state = 0):
        self._max_iter = max_iter
        self._tol = tol
        self._n_clusters = n_clusters
        self._seed = random_state
        self._labels = None
        self._centroids = None
        
    def _get_centroids_from_data(self, data):
        # to select k centroids from data
        np.random.seed(self._seed)
        selected_rows = np.random.randint(0, 
                                          self._get_num_observations(data), 
                                          size = self._n_clusters)
        self._centroids = np.array(data)[selected_rows, :]
        
    def _get_num_observations(self, data):
        return (np.array(data)).shape[0]

    def _get_num_features(self, data):
        # return the number of features based on input data.
        data = np.array(data)
        return data.shape[1]

    def _get_labels(self, data):
        # to obtain the labels in the dataset based on the input centroids
        self._labels = distance.cdist(data, self._centroids).argmin(axis = 1)

    def _get_centroids(self, data):
        labels = self._labels
        list_centroids = [np.average(data[labels == j, ], axis =0) 
            for j in range(self._n_clusters)]
        self._centroids = np.vstack(list_centroids)

    def _should_stop(self, old_centroids, iters):
        if iters > self._max_iter:
            return True
        else:
            if old_centroids is None:
                return False
            else:
                return (np.abs(old_centroids - self._centroids).sum() < self._tol)
        
    def fit(self, data):
        # initiating centroids, iteration, old_centroids 
        iters = 0
        old_centroids = self._centroids
        self._get_centroids_from_data(data)
        
        # running the k means
        while not self._should_stop(old_centroids, iters):
            # assigning
            old_centroids = self._centroids
            iters += 1
            
            # calculating the labels and centroids
            self._get_labels(data)
            self._get_centroids(data)
        