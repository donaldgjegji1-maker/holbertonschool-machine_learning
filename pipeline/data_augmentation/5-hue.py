import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
        image: A 3D tf.Tensor containing the image to change
        delta: The amount the hue should change (must be in the range [-1, 1])

    Returns:
        The hue-altered image as a tf.Tensor
    """
    return tf.image.adjust_hue(image, delta)
