#!/usr/bin/env python3
"""
Performs the Expectation-Maximization algorithm for a Gaussian Mixture Model.
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the EM algorithm for a Gaussian Mixture Model.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, d)
        Dataset.
    k : int
        Number of clusters.
    iterations : int
        Maximum iterations.
    tol : float
        Tolerance for early stopping based on log likelihood.
    verbose : bool
        If True, prints log likelihood during iterations.

    Returns
    -------
    pi : numpy.ndarray of shape (k,)
        Priors for each cluster.
    m : numpy.ndarray of shape (k, d)
        Centroid means for each cluster.
    S : numpy.ndarray of shape (k, d, d)
        Covariance matrices for each cluster.
    g : numpy.ndarray of shape (k, n)
        Probabilities for each point in each cluster.
    l : float
        Log likelihood of the model.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    n, d = X.shape
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    log_likelihood_prev = 0
    for i in range(iterations):
        g, l = expectation(X, pi, m, S)
        pi, m, S = maximization(X, g)

        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(f"Log Likelihood after {i} iterations: {l:.5f}")

        if abs(l - log_likelihood_prev) <= tol:
            break
        log_likelihood_prev = l

    return pi, m, S, g, l
