#!/usr/bin/env python3
"""
Module for same convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2

    pad_top = (kh - 1) // 2
    pad_bot = (kh - 1) - pad_top
    pad_left = (kw - 1) // 2
    pad_right = (kw - 1) - pad_left

    # Pad the images
    padded_images = np.pad(images,
                           ((0, 0), (pad_top, pad_bot), (pad_left, pad_right)),
                           mode='constant', constant_values=0)

    # Initialize output array
    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            region = padded_images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))
    return output
