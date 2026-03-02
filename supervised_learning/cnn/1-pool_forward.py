#!/usr/bin/env python3
"""Pooling Forward Propagation Module"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
        kernel_shape: tuple of (kh, kw) containing the size of the kernel
        stride: tuple of (sh, sw) containing the strides for the pooling
        mode: string containing either 'max' or 'avg'

    Returns:
        output of the pooling layer
    """
    # Get dimensions
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    h_out = int((h_prev - kh) / sh) + 1
    w_out = int((w_prev - kw) / sw) + 1

    # Initialize output
    A = np.zeros((m, h_out, w_out, c_prev))

    # Perform pooling
    for i in range(h_out):
        for j in range(w_out):
            # Calculate slice indices
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            # Extract slice from input
            A_slice = A_prev[:, h_start:h_end, w_start:w_end, :]

            if mode == 'max':
                # Max pooling: take maximum over height and width dimensions
                # Keep the channel dimension
                A[:, i, j, :] = np.max(A_slice, axis=(1, 2))

            elif mode == 'avg':
                # Average pooling: take mean over height and width dimensions
                A[:, i, j, :] = np.mean(A_slice, axis=(1, 2))

            else:
                raise ValueError("mode must be either 'max' or 'avg'")
    return A
