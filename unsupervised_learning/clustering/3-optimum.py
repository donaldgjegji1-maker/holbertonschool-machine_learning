#!/usr/bin/env python3
"""
Optimum K determination module for K-means clustering
"""

import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Determine the optimum number of clusters by analyzing variance differences.
    
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        kmin: positive integer, minimum number of clusters to check
        kmax: positive integer, maximum number of clusters to check
        iterations: positive integer, maximum iterations for K-means
        
    Returns:
        tuple: (results, d_vars) where:
            results: list of K-means outputs for each cluster size
            d_vars: list of variance differences from kmin
        or (None, None) on failure
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    n, d = X.shape
    if n == 0 or d == 0:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0 or kmin >= n:
        return None, None
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= kmin or kmax > n:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    
    # Import required modules
    kmeans = __import__('1-kmeans').kmeans
    variance = __import__('2-variance').variance
    
    results = []
    d_vars = []
    base_variance = None
    
    # Single loop to process all k values
    for k in range(kmin, kmax + 1):
        # Run K-means
        C, clss = kmeans(X, k, iterations)
        
        # Calculate variance
        var = variance(X, C)
        
        # Store results
        results.append((C, clss))
        
        # Calculate variance difference
        if k == kmin:
            base_variance = var
            d_vars.append(0.0)
        else:
            d_vars.append(base_variance - var)
    
    return results, d_vars
