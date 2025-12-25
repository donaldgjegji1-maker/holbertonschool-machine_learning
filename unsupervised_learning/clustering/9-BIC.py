#!/usr/bin/env python3
"""
Bayesian Information Criterion for Gaussian Mixture Models
"""

import numpy as np


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Find the best number of clusters using Bayesian Information Criterion.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        kmin: positive integer, minimum number of clusters to check
        kmax: positive integer, maximum number of clusters to check
        iterations: positive integer, max iterations for EM algorithm
        tol: non-negative float, tolerance for EM algorithm
        verbose: boolean, whether EM should print progress

    Returns:
        tuple: (best_k, best_result, l, b) where:
            best_k: best value for k based on BIC
            best_result: tuple (pi, m, S) for best_k
            l: numpy.ndarray of log likelihoods for each k
            b: numpy.ndarray of BIC values for each k
        or (None, None, None, None) on failure
    """
    # Validate inputs
    if (not all([
        isinstance(X, np.ndarray), len(X.shape) == 2,
        isinstance(kmin, int), kmin > 0, kmin <= X.shape[0],
        isinstance(iterations, int), iterations > 0,
        isinstance(tol, (int, float)), tol >= 0,
        isinstance(verbose, bool)
    ])):
        return None, None, None, None

    n, d = X.shape

    # Set kmax if None
    if kmax is None:
        kmax = n
    elif not isinstance(kmax, int) or kmax < kmin or kmax > n:
        return None, None, None, None

    # Import EM
    expectation_maximization = __import__('8-EM').expectation_maximization

    # Initialize results
    k_range = range(kmin, kmax + 1)
    num_ks = len(k_range)

    log_liks = np.zeros(num_ks)
    bic_vals = np.zeros(num_ks)
    all_results = []

    best_k_val = None
    best_result_val = None
    best_bic = float('inf')

    # Single loop over k values
    for idx, k in enumerate(k_range):
        # Run EM
        pi, m, S, _, lik = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:  # EM failed
            return None, None, None, None

        # Store results
        all_results.append((pi, m, S))
        log_liks[idx] = lik

        # Calculate number of parameters
        p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)

        # Calculate BIC
        bic = p * np.log(n) - 2 * lik
        bic_vals[idx] = bic

        if bic < best_bic:
            best_bic = bic
            best_k_val = k
            best_result_val = (pi, m, S)

    return best_k_val, best_result_val, log_liks, bic_vals
