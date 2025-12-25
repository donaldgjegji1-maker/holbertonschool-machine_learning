#!/usr/bin/env python3
"""
K-means clustering using scikit-learn
"""

import numpy as np
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
    # Input validation
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None

    # Initialize KMeans
    model = sklearn.cluster.KMeans(n_clusters=k, random_state=0)

    # Fit the model
    model.fit(X)

    # Get centroids and labels
    C = model.cluster_centers_
    labels = model.labels_

    sorted_indices = np.argsort(C[:, 0])
    C_sorted = C[sorted_indices]

    # Create a mapping from old labels to new sorted labels
    lm = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}

    # Update labels according to the new ordering
    labels_sorted = np.array([lm[label] for label in labels])

    return C_sorted, labels_sorted
