#!/usr/bin/env python3
"""
Gaussian Probability Density Function calculation module
"""

import numpy as np


def pdf(X, m, S):
    """
    Calculate the probability density function of a Gaussian distribution.

    Args:
        X: numpy.ndarray of shape (n, d) containing data points
        m: numpy.ndarray of shape (d,) containing mean of distribution
        S: numpy.ndarray of shape (d, d) containing covariance matrix

    Returns:
        numpy.ndarray of shape (n,) containing PDF values
        or None on failure
    """

    if (not isinstance(X, np.ndarray) or len(X.shape) != 2
            or not isinstance(m, np.ndarray) or len(m.shape) != 1
            or not isinstance(S, np.ndarray) or len(S.shape) != 2
            or X.shape[1] != m.shape[0]
            or X.shape[1] != S.shape[0]
            or S.shape[0] != S.shape[1]):

        return None

    n, d = X.shape

    try:
        det = np.linalg.det(S)
    except np.linalg.LinAlgError:
        return None

    if det <= 0:
        return None

    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return None

    # Normalization constant
    const = 1.0 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))

    diff = X - m
    quad_form = np.einsum('ni,ij,nj->n', diff, S_inv, diff)

    # Calculate PDF
    P = const * np.exp(-0.5 * quad_form)

    # Ensure minimum value
    P = np.maximum(P, 1e-300)

    return P
