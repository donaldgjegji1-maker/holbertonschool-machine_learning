#!/usr/bin/env python3
"""
F1 Score Calculation Module
"""

import numpy as np

# Import the required functions
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix.
    """
    recall = sensitivity(confusion)
    prec = precision(confusion)
    f1_scores = 2 * (prec * recall) / (prec + recall)

    return f1_scores
