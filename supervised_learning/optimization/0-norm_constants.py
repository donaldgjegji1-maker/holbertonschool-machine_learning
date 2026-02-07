#!/usr/bin/env python3
"""
Normalization Constants Module
"""

import numpy as np


def normalization_constants(X):
    """
    Calculate the normalization (standardization) constants of a matrix.
    """
    mean = np.mean(X, axis=0)

    variance = np.mean((X - mean) ** 2, axis=0)
    std = np.sqrt(variance)

    return mean, std
