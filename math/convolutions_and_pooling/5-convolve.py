#!/usr/bin/env python3
"""
Module for convolution on images with multiple kernels
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape

    if kc != c:
        raise ValueError("Kernel channels must match image channels")

    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2

        # Ensure padding is integer and handle edge cases
        ph = max(0, ((h - 1) * sh + kh - h + 1) // 2)
        pw = max(0, ((w - 1) * sw + kw - w + 1) // 2)

    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded_images = np.pad(images,
                           ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)

    padded_h = h + 2 * ph
    padded_w = w + 2 * pw
    output_h = (padded_h - kh) // sh + 1
    output_w = (padded_w - kw) // sw + 1

    output = np.zeros((m, output_h, output_w, nc))

    for i in range(output_h):
        for j in range(output_w):
            start_i = i * sh
            start_j = j * sw

            # Extract region
            r = padded_images[:, start_i:start_i+kh, start_j:start_j+kw, :]

            for k in range(nc):
                current_kernel = kernels[:, :, :, k]
                output[:, i, j, k] = np.sum(r * current_kernel, axis=(1, 2, 3))

    return output
