#!/usr/bin/env python3
"""
Agglomerative clustering with Ward linkage
"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt
import numpy as np


def agglomerative(X, dist):
    """
    Perform agglomerative clustering with Ward linkage.
    
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        dist: maximum cophenetic distance for all clusters
        
    Returns:
        clss: numpy.ndarray of shape (n,) containing cluster indices
    """
    # Validate inputs
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
        not isinstance(dist, (int, float)) or dist <= 0):
        return None
    
    n = X.shape[0]
    
    # Perform hierarchical clustering with Ward linkage
    # Ward linkage: minimizes the variance of the clusters being merged
    Z = scipy.cluster.hierarchy.linkage(X, method='ward', metric='euclidean')
    
    # Form flat clusters by cutting the dendrogram at distance 'dist'
    # fcluster returns cluster assignments (1-indexed)
    cluster_assignments = scipy.cluster.hierarchy.fcluster(
        Z, t=dist, criterion='distance'
    )
    
    # Convert to 0-indexed clusters
    clss = cluster_assignments - 1
    
    # Calculate cophenetic distances
    # The cophenetic distance between two objects is the height
    # of the dendrogram where they become members of the same cluster
    coph_dist = scipy.cluster.hierarchy.cophenet(Z)
    
    # Create and display the dendrogram
    plt.figure(figsize=(12, 8))
    
    # Set up dendrogram parameters
    dendro_params = {
        'Z': Z,
        'color_threshold': dist,  # Color clusters below this distance
        'above_threshold_color': 'k',  # Color above threshold in black
        'truncate_mode': 'lastp',
        'p': min(30, n),  # Show last p merged clusters, max 30
        'show_leaf_counts': True,
        'leaf_rotation': 90.,
        'leaf_font_size': 10.,
        'show_contracted': True  # Contract dense branches
    }
    
    # Plot dendrogram
    scipy.cluster.hierarchy.dendrogram(**dendro_params)
    
    # Add cutoff line and labels
    plt.axhline(y=dist, color='red', linestyle='--', 
                linewidth=2, label=f'Cutoff: {dist}')
    plt.title(f'Agglomerative Clustering Dendrogram (Ward linkage)\n'
              f'Cutoff distance: {dist}, Number of clusters: {len(np.unique(clss))}')
    plt.xlabel('Sample index (or cluster size)')
    plt.ylabel('Distance (Ward linkage)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return clss