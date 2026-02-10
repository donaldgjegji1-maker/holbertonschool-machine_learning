#!/usr/bin/env python3
"""
Sensitivity (Recall) Calculation Module
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.
    """
    # Sensitivity = TP / (TP + FN)
    # For each class i (row i):
    # - TP (true positives) = confusion[i, i]
    # - FN (false negatives) = sum(confusion[i, :]) - TP

    classes = confusion.shape[0]
    sensitivity_scores = np.zeros(classes)

    for i in range(classes):
        tp = confusion[i, i]
        actual_positives = np.sum(confusion[i, :])

        # Sensitivity = TP / (TP + FN) = TP / actual_positives
        sensitivity_scores[i] = tp / actual_positives

    return sensitivity_scores
