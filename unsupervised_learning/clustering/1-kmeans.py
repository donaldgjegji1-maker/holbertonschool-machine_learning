#!/usr/bin/env python3
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
    
    # First use of np.random.uniform: initialize centroids
    # Get min and max for each dimension
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    
    # Initialize centroids uniformly within data bounds
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))
    
    # Store previous centroids to check for convergence
    prev_C = np.copy(C)
    
    for i in range(iterations):
        # Assign each data point to nearest centroid (E-step)
        # Using broadcasting to compute distances
        distances = np.linalg.norm(X[:, np.newaxis, :] - C[np.newaxis, :, :], axis=2)
        clss = np.argmin(distances, axis=1)
        
        # Update centroids (M-step)
        for j in range(k):
            # Get points assigned to cluster j
            cluster_points = X[clss == j]
            
            if len(cluster_points) > 0:
                # Update centroid as mean of assigned points
                C[j] = cluster_points.mean(axis=0)
            else:
                # Reinitialize empty cluster centroid
                C[j] = np.random.uniform(low=min_vals, high=max_vals)
        
        # Check for convergence
        if np.allclose(prev_C, C):
            break
            
        prev_C = np.copy(C)
    
    # One final assignment
    distances = np.linalg.norm(X[:, np.newaxis, :] - C[np.newaxis, :, :], axis=2)
    clss = np.argmin(distances, axis=1)
    
    return C, clss
