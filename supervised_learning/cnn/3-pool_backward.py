#!/usr/bin/env python3
"""Pooling Backward Propagation Module"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network

    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c)
            containing partial derivatives with respect to
            output of pooling layer
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c)
            containing output of previous layer
        kernel_shape: tuple of (kh, kw) containing kernel size
        stride: tuple of (sh, sw) containing strides for pooling
        mode: string containing either 'max' or 'avg'

    Returns:
        dA_prev: partial derivatives with respect to previous layer
    """
    # Get dimensions
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize dA_prev with zeros
    dA_prev = np.zeros_like(A_prev)

    # Loop over all examples
    for i in range(m):
        # Loop over output height
        for h in range(h_new):
            # Loop over output width
            for w in range(w_new):
                # Loop over channels
                for ch in range(c):
                    # Calculate slice indices in input
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    if mode == 'max':
                        # Get the slice from A_prev
                        a_slice = A_prev[
                            i, h_start:h_end, w_start:w_end, ch
                        ]

                        # Create mask for the maximum value
                        mask = (a_slice == np.max(a_slice))

                        # Distribute gradient to the max position
                        dA_prev[
                            i, h_start:h_end, w_start:w_end, ch
                        ] += mask * dA[i, h, w, ch]

                    elif mode == 'avg':
                        # Calculate the average gradient contribution
                        avg_grad = dA[i, h, w, ch] / (kh * kw)

                        # Distribute gradient equally to all positions
                        dA_prev[
                            i, h_start:h_end, w_start:w_end, ch
                        ] += np.ones((kh, kw)) * avg_grad

                    else:
                        raise ValueError(
                            "mode must be either 'max' or 'avg'"
                        )

    return dA_prev
