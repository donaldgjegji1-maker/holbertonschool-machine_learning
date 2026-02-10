#!/usr/bin/env python3
"""
Confusion Matrix Creation Module
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.
    """
    # Convert one-hot to class indices
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))

    for true, pred in zip(true_classes, pred_classes):
        confusion[true, pred] += 1

    return confusion
