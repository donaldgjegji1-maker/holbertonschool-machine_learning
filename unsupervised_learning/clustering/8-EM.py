#!/usr/bin/env python3
"""
Performs the Expectation-Maximization algorithm GMM.
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the EM algorithm.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, d)
        Dataset.
    k : int
        Number of clusters.
    iterations : int
        Maximum iterations.
    tol : float
        Tolerance to determine early stopping.
    verbose : bool
        If True, prints log likelihood during iterations.

    Returns
    -------
    pi : numpy.ndarray of shape (k,)
        Priors.
    m : numpy.ndarray of shape (k, d)
        Centroid means.
    S : numpy.ndarray of shape (k, d, d)
        Covariance matrices.
    g : numpy.ndarray of shape (k, n)
        Probabilities.
    l : float
        Log likelihood of the model.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    n, d = X.shape

    if k > n:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    log_likelihood_prev = 0

    for i in range(iterations):
        g, log_l = expectation(X, pi, m, S)

        if g is None or log_l is None:
            return None, None, None, None, None

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {:.5f}".format(
                i, log_l))

        if i > 0 and abs(log_l - log_likelihood_prev) <= tol:
            if verbose:
                print("Log Likelihood after {} iterations: {:.5f}".format(
                    i, log_l))
            break

        log_likelihood_prev = log_l

        pi, m, S = maximization(X, g)

        if pi is None or m is None or S is None:
            return None, None, None, None, None
    else:
        # Final expectation step after last iteration
        g, log_l = expectation(X, pi, m, S)

        if g is None or log_l is None:
            return None, None, None, None, None

        if verbose:
            print("Log Likelihood after {} iterations: {:.5f}".format(
                iterations, log_l))

    return pi, m, S, g, log_l
