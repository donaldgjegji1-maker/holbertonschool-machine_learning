#!/usr/bin/env python3
"""
Gaussian Mixture Model using scikit-learn
"""

import sklearn.mixture


def gmm(X, k):
    """
    Calculate Gaussian Mixture Model using sklearn.
    
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: integer, number of clusters
        
    Returns:
        tuple: (pi, m, S, clss, bic) where:
            pi: numpy.ndarray of shape (k,) containing cluster priors
            m: numpy.ndarray of shape (k, d) containing centroid means
            S: numpy.ndarray of shape (k, d, d) containing covariance matrices
            clss: numpy.ndarray of shape (n,) containing cluster indices
            bic: float containing the BIC value
    """
    # Initialize and fit GMM
    model = sklearn.mixture.GaussianMixture(
        n_components=k,
        covariance_type='full',
        random_state=0
    )
    
    # Fit the model
    model.fit(X)
    
    # Get parameters
    pi = model.weights_          # Mixing coefficients (priors)
    m = model.means_             # Means of each Gaussian component
    S = model.covariances_       # Covariance matrices of each component
    
    # Get cluster assignments (hard clustering)
    clss = model.predict(X)
    
    # Calculate BIC
    bic = model.bic(X)
    
    return pi, m, S, clss, bic
