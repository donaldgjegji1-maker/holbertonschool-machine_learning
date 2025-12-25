#!/usr/bin/env python3
"""
K-means clustering using scikit-learn
"""

import sklearn.cluster


def kmeans(X, k):
    """
    Perform K-means clustering using sklearn.

    Args:
        X: numpy.ndarray of shape (n, d) containing dataset
        k: int, number of clusters

    Returns:
        C: numpy.ndarray of shape (k, d), centroids
        clss: numpy.ndarray of shape (n,), cluster labels
    """
    # Create and fit KMeans model
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)

    # Get centroids and labels
    C = model.cluster_centers_
    clss = model.labels_

    return C, clss
