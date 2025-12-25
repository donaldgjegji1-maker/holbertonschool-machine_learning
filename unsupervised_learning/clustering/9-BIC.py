#!/usr/bin/env python3
"""
Bayesian Information Criterion GMM
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Find the best number of clusters using BIC.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        kmin: positive integer, minimum number of clusters to check
        kmax: positive integer, maximum number of clusters to check
        iterations: positive integer, max iterations
        tol: non-negative float, tolerance
        verbose: boolean, whether to print progress

    Returns:
        tuple: (best_k, best_result, l, b) where:
            best_k: best value based on BIC
            best_result: tuple (pi, m, S)
            l: numpy.ndarray of log likelihoods
            b: numpy.ndarray of BIC values
        or (None, None, None, None) on failure
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    # Set kmax if None
    if kmax is None:
        kmax = n

    # Validate kmax
    if not isinstance(kmax, int) or kmax < 1:
        return None, None, None, None

    # Check kmax >= kmin
    if kmax < kmin:
        return None, None, None, None

    # Cap kmax at n if needed
    if kmax > n:
        kmax = n

    # Check kmin is valid
    if kmin > n:
        return None, None, None, None

    # Initialize results
    k_range = range(kmin, kmax + 1)
    num_ks = len(k_range)

    log_liks = np.zeros(num_ks)
    bic_vals = np.zeros(num_ks)

    best_k_val = None
    best_result_val = None
    best_bic = float('inf')

    # Single loop over k values
    for idx, k in enumerate(k_range):
        # Run EM
        pi, m, S, _, log_l = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return None, None, None, None

        # Store log likelihood
        log_liks[idx] = log_l

        # Calculate number of parameters
        # pi: k-1 (since they sum to 1)
        # m: k*d (mean vectors)
        # S: k*d*(d+1)/2 (symmetric covariance matrices)
        p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)

        # Calculate BIC: p * ln(n) - 2 * l
        bic = p * np.log(n) - 2 * log_l
        bic_vals[idx] = bic

        # Track best result (lowest BIC)
        if bic < best_bic:
            best_bic = bic
            best_k_val = k
            best_result_val = (pi, m, S)

    return best_k_val, best_result_val, log_liks, bic_vals
