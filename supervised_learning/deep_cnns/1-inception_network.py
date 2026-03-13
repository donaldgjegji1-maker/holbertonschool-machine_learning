#!/usr/bin/env python3
"""Inception Network"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the inception network (GoogLeNet).

    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))

    # Stem
    x = K.layers.Conv2D(64, 7, strides=2, padding='same',
                        activation='relu')(X)
    x = K.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    x = K.layers.Conv2D(64, 1, padding='same', activation='relu')(x)
    x = K.layers.Conv2D(192, 3, padding='same', activation='relu')(x)
    x = K.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Inception 3a, 3b
    x = inception_block(x, [64,  96,  128, 16, 32,  32])
    x = inception_block(x, [128, 128, 192, 32, 96,  64])
    x = K.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Inception 4a, 4b, 4c, 4d, 4e
    x = inception_block(x, [192, 96,  208, 16, 48,  64])
    x = inception_block(x, [160, 112, 224, 24, 64,  64])
    x = inception_block(x, [128, 128, 256, 24, 64,  64])
    x = inception_block(x, [112, 144, 288, 32, 64,  64])
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = K.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Inception 5a, 5b
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = inception_block(x, [384, 192, 384, 48, 128, 128])

    # Classifier head
    x = K.layers.AveragePooling2D(7, strides=1)(x)
    x = K.layers.Dropout(0.4)(x)
    x = K.layers.Dense(1000, activation='softmax')(x)

    return K.models.Model(inputs=X, outputs=x)
