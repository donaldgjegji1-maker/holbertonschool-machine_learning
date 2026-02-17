#!/usr/bin/env python3
"""
Function that performs a valid convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.
    """
    # Extract dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate output dimensions:
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize output array with zeros
    # Shape: (m, output_h, output_w)
    output = np.zeros((m, output_h, output_w))

    # Slide the kernel over every valid position
    # Loop 1: iterate over output height positions
    for i in range(output_h):
        # Loop 2: iterate over output width positions
        for j in range(output_w):
            # Extract the region from ALL images at once
            region = images[:, i:i + kh, j:j + kw]

            # Multiply region with kernel, then sum
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
