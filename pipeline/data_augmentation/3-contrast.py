import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        image: A 3D tf.Tensor representing
        the input image to adjust the contrast

        lower: A float representing
        the lower bound of the random contrast factor range

        upper: A float representing
        the upper bound of the random contrast factor range

    Returns:
        The contrast-adjusted image as a tf.Tensor
    """
    # Generate a random contrast factor between lower and upper
    contrast_factor = tf.random.uniform([], lower, upper)

    return tf.image.adjust_contrast(image, contrast_factor)
