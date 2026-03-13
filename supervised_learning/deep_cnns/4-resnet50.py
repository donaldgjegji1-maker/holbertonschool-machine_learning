#!/usr/bin/env python3
"""ResNet-50"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture.

    Returns: the keras model
    """
    init = K.initializers.HeNormal(seed=0)
    X = K.Input(shape=(224, 224, 3))

    # Stem
    x = K.layers.Conv2D(64, 7, strides=2, padding='same',
                        kernel_initializer=init)(X)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Stage 2: 1 projection + 2 identity  (56x56, depth 256)
    x = projection_block(x, [64,  64,  256],  s=1)
    x = identity_block(x,   [64,  64,  256])
    x = identity_block(x,   [64,  64,  256])

    # Stage 3: 1 projection + 3 identity  (28x28, depth 512)
    x = projection_block(x, [128, 128, 512],  s=2)
    x = identity_block(x,   [128, 128, 512])
    x = identity_block(x,   [128, 128, 512])
    x = identity_block(x,   [128, 128, 512])

    # Stage 4: 1 projection + 5 identity  (14x14, depth 1024)
    x = projection_block(x, [256, 256, 1024], s=2)
    x = identity_block(x,   [256, 256, 1024])
    x = identity_block(x,   [256, 256, 1024])
    x = identity_block(x,   [256, 256, 1024])
    x = identity_block(x,   [256, 256, 1024])
    x = identity_block(x,   [256, 256, 1024])

    # Stage 5: 1 projection + 2 identity  (7x7, depth 2048)
    x = projection_block(x, [512, 512, 2048], s=2)
    x = identity_block(x,   [512, 512, 2048])
    x = identity_block(x,   [512, 512, 2048])

    # Classifier head
    x = K.layers.AveragePooling2D(7, strides=1)(x)
    x = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=init)(x)

    return K.models.Model(inputs=X, outputs=x)
