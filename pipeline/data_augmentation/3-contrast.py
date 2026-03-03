#!/usr/bin/env python3
"""Module for randomly adjusting image contrast"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly adjusts the contrast of an image.

    Args:
        image: a 3D tf.Tensor representing the input image

        lower: a float representing the lower
        bound of the contrast factor range

        upper: a float representing the upper
        bound of the contrast factor range

    Returns:
        The contrast-adjusted image
    """
    return tf.image.random_contrast(image, lower, upper)
