#!/usr/bin/env python3
"""
K-means clustering using scikit-learn
"""

import sklearn.cluster


def kmeans(X, k):
    """
    Perform K-means clustering using sklearn's KMeans.
    
    Args:
        X: numpy.ndarray of shape (n, d) containing dataset
        k: int, number of clusters
        
    Returns:
        C: numpy.ndarray of shape (k, d), centroids
        clss: numpy.ndarray of shape (n,), cluster labels
    """
    # Initialize KMeans
    model = sklearn.cluster.KMeans(n_clusters=k, random_state=0)
    
    # Fit the model
    model.fit(X)
    
    # Return centroids and labels
    return model.cluster_centers_, model.labels_
