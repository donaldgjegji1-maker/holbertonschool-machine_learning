#!/usr/bin/env python3
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization

if __name__ == "__main__":
    np.random.seed(11)
    
    # Create synthetic data from 4 Gaussian clusters
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    
    k = 4
    pi, m, S, g, l = expectation_maximization(X, k, 150, verbose=True)
    
    # Assign each point to the cluster with highest responsibility
    clss = np.sum(g * np.arange(k).reshape(k, 1), axis=0)
    
    # Print results
    print(X.shape[0] * pi)  # Expected number of points per cluster
    print(m)                # Final cluster means
    print(S)                # Final covariance matrices
    print(l)                # Final log likelihood
