#!/usr/bin/env python3
"""
Performs K-means on a dataset
"""

import numpy as np

def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.
    
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, number of clusters
        iterations: positive integer, maximum number of iterations
    
    Returns:
        C, clss or None, None on failure
        C: numpy.ndarray of shape (k, d) containing centroid means
        clss: numpy.ndarray of shape (n,) containing cluster indices
    """
    # Input validation
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    
    n, d = X.shape
    
    # Get data bounds
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    
    # FIRST use of np.random.uniform: initialize centroids
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))
    
    # SECOND use of np.random.uniform: pre-generate random centroids
    # for potential empty cluster reinitialization
    random_centroids = np.random.uniform(low=min_vals, high=max_vals, 
                                         size=(k * iterations, d))
    random_idx = 0
    
    # Main algorithm
    for _ in range(iterations):
        C_prev = np.copy(C)
        
        # Assign points to centroids (vectorized)
        diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        clss = np.argmin(distances, axis=1)
        
        # Update centroids (ONE loop)
        for j in range(k):
            cluster_points = X[clss == j]
            
            if len(cluster_points) > 0:
                C[j] = cluster_points.mean(axis=0)
            else:
                # Use next random centroid from pre-generated pool
                C[j] = random_centroids[random_idx]
                random_idx += 1
        
        # Check convergence
        if np.allclose(C_prev, C, atol=1e-8):
            break
    
    # Final assignment
    diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    clss = np.argmin(distances, axis=1)
    
    return C, clss
