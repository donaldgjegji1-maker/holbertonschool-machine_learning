#!/usr/bin/env python3
"""
Module for convolution on grayscale images with padding and stride options
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a conv on grayscale images with configurable padding and stride
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
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

    # Pad the images
    padded_images = np.pad(images,
                           ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    # Calculate output dimensions
    padded_h = h + 2 * ph
    padded_w = w + 2 * pw

    output_h = (padded_h - kh) // sh + 1
    output_w = (padded_w - kw) // sw + 1

    # Initialize output array
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            # Calculate the starting position in the padded image
            start_i = i * sh
            start_j = j * sw
            region = padded_images[:, start_i:start_i+kh, start_j:start_j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
