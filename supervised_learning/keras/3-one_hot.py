#!/usr/bin/env python3
"""
Convert labels to one-hot encoding
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Convert a label vector into a one-hot matrix
    """
    if classes is None:
        max_value = K.backend.max(K.backend.constant(labels))
        classes = int(K.backend.eval(max_value)) + 1

    result = K.backend.one_hot(
        K.backend.cast(K.backend.constant(labels), 'int32'),
        classes
    )

    return K.backend.eval(result)
