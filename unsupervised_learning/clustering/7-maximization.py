#!/usr/bin/env python3
"""
Maximization step for Gaussian Mixture Model EM algorithm
"""

import numpy as np


def maximization(X, g):
    """
    Calculate the maximization step in EM algorithm for GMM.
    
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        g: numpy.ndarray of shape (k, n) containing posterior probabilities
        
    Returns:
        tuple: (pi, m, S) where:
            pi: numpy.ndarray of shape (k,) with updated priors
            m: numpy.ndarray of shape (k, d) with updated means
            S: numpy.ndarray of shape (k, d, d) with updated covariances
        or (None, None, None) on failure
    """
    # Input validation
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
        not isinstance(g, np.ndarray) or len(g.shape) != 2):
        return None, None, None
    
    n, d = X.shape
    k = g.shape[0]
    
    if g.shape[1] != n:
        return None, None, None
    
    # Calculate N_k (effective number of points in each cluster)
    N_k = np.sum(g, axis=1)  # shape (k,)
    
    # Avoid division by zero
    N_k_safe = np.maximum(N_k, 1e-15)
    
    # Update priors: π_k = N_k / N
    pi = N_k / n
    
    # Update means: μ_k = (Σ_n γ_nk x_n) / N_k
    # Vectorized calculation using matrix multiplication
    # g has shape (k, n), X has shape (n, d)
    # Result: (k, n) @ (n, d) = (k, d)
    m = (g @ X) / N_k_safe[:, np.newaxis]
    
    # Initialize covariance matrices
    S = np.zeros((k, d, d))
    
    # Update covariance matrices (using one loop)
    for i in range(k):
        # Differences from mean
        diff = X - m[i]  # shape (n, d)
        
        # Weight by responsibilities
        weighted_diff = np.sqrt(g[i, :, np.newaxis]) * diff  # shape (n, d)
        
        # Covariance = weighted_diff^T @ weighted_diff / N_k
        # Equivalent to Σ_n γ_nk (x_n - μ_k)(x_n - μ_k)^T / N_k
        S[i] = (weighted_diff.T @ weighted_diff) / N_k_safe[i]
        
        # Add small regularization to ensure positive definite
        S[i] += np.eye(d) * 1e-6
    
    return pi, m, S
