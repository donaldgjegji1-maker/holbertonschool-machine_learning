#!/usr/bin/env python3
"""
Precision Calculation Module
"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.
    """
    # Precision = TP / (TP + FP)
    # For each class i (column i):
    # - TP (true positives) = confusion[i, i]
    # - FP (false positives) = sum(confusion[:, i]) - TP

    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)

    # Precision = TP / (TP + FP) = TP / predicted_positives
    precision_scores = true_positives / predicted_positives
    return precision_scores
