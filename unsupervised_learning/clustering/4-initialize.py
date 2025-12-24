#!/usr/bin/env python3
"""
Gaussian Mixture Model initialization module
"""

import numpy as np


def initialize(X, k):
    """
    Initialize variables for a Gaussian Mixture Model.
    
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters
        
    Returns:
        tuple: (pi, m, S) where:
            pi: numpy.ndarray of shape (k,) with priors for each cluster
            m: numpy.ndarray of shape (k, d) with centroid means
            S: numpy.ndarray of shape (k, d, d) with covariance matrices
        or (None, None, None) on failure
    """
    # Input validation
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
        not isinstance(k, int) or k <= 0 or k > X.shape[0]):
        return None, None, None
    
    try:
        # Import K-means
        kmeans = __import__('1-kmeans').kmeans
        
        # Initialize priors: equal probability for each cluster
        pi = np.full(k, 1.0 / k)
        
        # Get centroids from K-means
        m, _ = kmeans(X, k, 1000)  # Default to 1000 iterations
        
        # Create k identity matrices of shape (d, d)
        d = X.shape[1]
        S = np.repeat(np.eye(d)[np.newaxis, :, :], k, axis=0)
        
        return pi, m, S
        
    except Exception:
        return None, None, None
