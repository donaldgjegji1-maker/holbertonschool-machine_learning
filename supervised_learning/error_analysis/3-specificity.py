#!/usr/bin/env python3
"""
Specificity Calculation Module
"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.
    """
    classes = confusion.shape[0]
    specificity_scores = np.zeros(classes)
    for i in range(classes):
        # True negatives: all elements except row i and column i
        total = np.sum(confusion)
        row_sum = np.sum(confusion[i, :])
        col_sum = np.sum(confusion[:, i])

        true_negatives = total - row_sum - col_sum + confusion[i, i]

        # False positives: sum of column i minus true positives
        false_positives = col_sum - confusion[i, i]

        # Specificity = TN / (TN + FP)
        specificity_scores[i] = true_negatives/(true_negatives+false_positives)

    return specificity_scores
