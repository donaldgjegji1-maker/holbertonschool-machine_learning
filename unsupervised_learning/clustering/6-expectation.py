#!/usr/bin/env python3
"""
Expectation step in EM algorithm GMM
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculate the expectation step in EM algorithm.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        pi: numpy.ndarray of shape (k,) containing priors
        m: numpy.ndarray of shape (k, d) containing centroid means
        S: numpy.ndarray of shape (k, d, d) containing covariance matrices

    Returns:
        tuple: (g, l) where:
            g: numpy.ndarray of shape (k, n) with posterior probabilities
            l: total log likelihood
        or (None, None) on failure
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    # Calculate PDF values
    likelihoods = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])
        if P is None:
            return None, None
        likelihoods[i] = P

    weighted = pi[:, np.newaxis] * likelihoods

    # Marginal probability
    marginal = np.sum(weighted, axis=0, keepdims=True)

    # Posterior probabilities
    g = weighted / marginal

    likelihood = np.sum(np.log(marginal))

    return g, likelihood
