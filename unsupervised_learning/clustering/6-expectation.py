#!/usr/bin/env python3
"""
Expectation step for Gaussian Mixture Model EM algorithm
"""

import numpy as np


def expectation(X, pi, m, S):
    """
    Calculate the expectation step in EM algorithm for GMM.
    
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        pi: numpy.ndarray of shape (k,) containing priors for each cluster
        m: numpy.ndarray of shape (k, d) containing centroid means
        S: numpy.ndarray of shape (k, d, d) containing covariance matrices
        
    Returns:
        tuple: (g, l) where:
            g: numpy.ndarray of shape (k, n) with posterior probabilities
            l: total log likelihood
        or (None, None) on failure
    """
    # Input validation
    if not all([isinstance(arr, np.ndarray) for arr in [X, pi, m, S]]):
        return None, None
    
    n, d = X.shape
    k = pi.shape[0]
    
    if (X.shape[1] != d or m.shape != (k, d) or S.shape != (k, d, d)):
        return None, None
    
    # Import pdf function
    pdf = __import__('5-pdf').pdf
    
    # Calculate PDFs for all clusters
    # Using list comprehension (considered as part of the loop)
    pdfs = np.array([pdf(X, m[i], S[i]) for i in range(k)])
    
    # Weighted likelihoods
    weighted = pi.reshape(-1, 1) * pdfs
    
    # Sum over clusters (denominator)
    sum_weighted = np.sum(weighted, axis=0, keepdims=True)
    
    # Add small value to avoid division by zero
    sum_weighted = np.clip(sum_weighted, 1e-300, np.inf)
    
    # Posterior probabilities
    g = weighted / sum_weighted
    
    # Log likelihood
    l = np.sum(np.log(sum_weighted))
    
    return g, l
