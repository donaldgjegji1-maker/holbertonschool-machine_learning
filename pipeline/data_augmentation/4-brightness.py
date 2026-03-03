import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Args:
        image: A 3D tf.Tensor containing the image to change
        max_delta: The max amount the image should be brightened (or darkened)

    Returns:
        The brightness-altered image as a tf.Tensor
    """
    return tf.image.random_brightness(image, max_delta)
