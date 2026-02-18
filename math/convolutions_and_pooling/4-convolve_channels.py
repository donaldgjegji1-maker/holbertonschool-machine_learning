#!/usr/bin/env python3
"""
Module for convolution on images with channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with multiple channels
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape

    if kc != c:
        raise ValueError("Kernel channels must match image channels")

    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
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

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            # Calculate the starting position in the padded image
            start_i = i * sh
            start_j = j * sw

            reg = padded_images[:, start_i:start_i+kh, start_j:start_j+kw, :]
            output[:, i, j] = np.sum(reg * kernel, axis=(1, 2, 3))

    return output
