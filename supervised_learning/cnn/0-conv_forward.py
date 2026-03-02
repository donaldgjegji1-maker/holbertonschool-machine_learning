#!/usr/bin/env python3
"""Convolutional Forward Propagation Module"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network
    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
            containing the kernels for the convolution
        b: numpy.ndarray of shape (1, 1, 1, c_new)
            containing the biases applied to the convolution
        activation: activation function applied to the convolution
        padding: string that is either 'same' or 'valid'
        stride: tuple of (sh, sw) containing the strides for the convolution

    Returns:
        output of the convolutional layer
    """
    # Get dimensions
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev_k, c_new = W.shape
    # Verify that c_prev matches between input and kernels
    if c_prev != c_prev_k:
        raise ValueError(f"Nr. of channels in input ({c_prev}) does not match "
                         f"nr. of channels in kernels ({c_prev_k})")

    sh, sw = stride

    # Calculate output dimensions based on padding type
    if padding == 'same':
        # Calculate padding needed for 'same' convolution
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1

        # Apply padding
        A_prev_padded = np.pad(
            A_prev,
            pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
            mode='constant',
            constant_values=0
        )

        # Calculate output dimensions after padding
        h_out = h_prev
        w_out = w_prev
    elif padding == 'valid':
        # No padding
        ph, pw = 0, 0
        A_prev_padded = A_prev

        # Calculate output dimensions for 'valid' convolution
        h_out = int((h_prev - kh) / sh) + 1
        w_out = int((w_prev - kw) / sw) + 1

    else:
        raise ValueError("padding must be either 'same' or 'valid'")

    # Initialize output
    Z = np.zeros((m, h_out, w_out, c_new))

    # Perform convolution
    for i in range(h_out):
        for j in range(w_out):
            for k in range(c_new):
                # Calculate slice indices
                h_start = i * sh
                h_end = h_start + kh
                w_start = j * sw
                w_end = w_start + kw

                # Extract slice from padded input
                A_slice = A_prev_padded[:, h_start:h_end, w_start:w_end, :]
                # Apply convolution: element-wise multiplication and sum
                # W[:, :, :, k] has shape (kh, kw, c_prev)
                # A_slice has shape (m, kh, kw, c_prev)
                # We need to multiply and sum over axes 1,2,3
                Z[:, i, j, k] = np.sum(A_slice * W[:, :, :, k], axis=(1, 2, 3))

    # Add bias
    Z = Z + b

    # Apply activation function
    A = activation(Z)
    return A
