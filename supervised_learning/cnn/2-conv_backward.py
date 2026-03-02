#!/usr/bin/env python3
"""Convolutional Backward Propagation Module"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer

    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new)
            containing partial derivatives
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing output of previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
            containing kernels for convolution
        b: numpy.ndarray of shape (1, 1, 1, c_new)
            containing biases applied to convolution
        padding: string that is either 'same' or 'valid'
        stride: tuple of (sh, sw) containing strides

    Returns:
        dA_prev: partial derivatives wrt previous layer
        dW: partial derivatives wrt kernels
        db: partial derivatives wrt biases
    """
    # Get dimensions
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev_k, c_new_k = W.shape
    sh, sw = stride

    # Verify dimensions match
    if c_prev != c_prev_k:
        raise ValueError(
            f"Channels in A_prev ({c_prev}) != "
            f"channels in W ({c_prev_k})"
        )
    if c_new != c_new_k:
        raise ValueError(
            f"Channels in dZ ({c_new}) != "
            f"channels in W ({c_new_k})"
        )

    # Initialize gradients
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Handle padding
    if padding == 'same':
        # Calculate padding for 'same' convolution
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1

        # Pad A_prev and dA_prev
        A_prev_padded = np.pad(
            A_prev,
            pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
            mode='constant',
            constant_values=0
        )
        dA_prev_padded = np.pad(
            dA_prev,
            pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
            mode='constant',
            constant_values=0
        )
    elif padding == 'valid':
        ph, pw = 0, 0
        A_prev_padded = A_prev
        dA_prev_padded = dA_prev
    else:
        raise ValueError("padding must be either 'same' or 'valid'")

    # Perform backpropagation
    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                # Calculate slice indices in padded input
                h_start = i * sh
                h_end = h_start + kh
                w_start = j * sw
                w_end = w_start + kw

                # Extract slice from padded A_prev
                A_slice = A_prev_padded[
                    :, h_start:h_end, w_start:w_end, :
                ]

                # Reshape dZ for broadcasting
                dz_reshaped = dZ[:, i, j, k].reshape(
                    (m, 1, 1, 1)
                )

                # Add to dA_prev_padded
                dA_prev_padded[
                    :, h_start:h_end, w_start:w_end, :
                ] += W[:, :, :, k] * dz_reshaped

                # Add to dW
                dW[:, :, :, k] += np.sum(
                    A_slice * dz_reshaped,
                    axis=0
                )

    # Remove padding from dA_prev if necessary
    if padding == 'same':
        dA_prev = dA_prev_padded[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
