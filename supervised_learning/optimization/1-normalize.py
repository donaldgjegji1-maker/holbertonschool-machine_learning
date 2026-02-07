#!/usr/bin/env python3
"""
Normalization Module
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.
    """
    X_normalized = (X - m) / s
    return X_normalized
