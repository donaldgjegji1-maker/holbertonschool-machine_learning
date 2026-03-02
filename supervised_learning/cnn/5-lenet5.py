#!/usr/bin/env python3
"""LeNet-5 Architecture Implementation using Keras"""

from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using Keras

    Args:
        X: K.Input of shape (m, 28, 28, 1) containing input images

    Returns:
        K.Model compiled with Adam optimization and accuracy metrics
    """
    # He normal initializer with seed=0 for reproducibility
    initializer = K.initializers.HeNormal(seed=0)

    # Layer 1: Convolutional layer with 6 kernels of shape 5x5, same padding
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=initializer
    )(X)

    # Layer 2: Max pooling layer with 2x2 kernel and 2x2 strides
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # Layer 3: Convolutional layer with 16 kernels of shape 5x5, valid padding
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=initializer
    )(pool1)

    # Layer 4: Max pooling layer with 2x2 kernel and 2x2 strides
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten before fully connected layers
    flatten = K.layers.Flatten()(pool2)

    # Layer 5: Fully connected layer with 120 nodes
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=initializer
    )(flatten)

    # Layer 6: Fully connected layer with 84 nodes
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=initializer
    )(fc1)

    # Layer 7: Fully connected softmax output layer with 10 nodes
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=initializer
    )(fc2)

    # Create model
    model = K.models.Model(inputs=X, outputs=output)

    # Compile model with Adam optimizer and accuracy metrics
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
