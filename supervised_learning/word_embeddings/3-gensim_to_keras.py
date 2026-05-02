#!/usr/bin/env python3
"""Converts a gensim word2vec model to a keras Embedding layer"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    """
    keys = model.wv
    weights = keys.vectors

    return tf.keras.layers.Embedding(input_dim=weights.shape[0],
                                     output_dim=weights.shape[1],
                                     weights=[weights],
                                     trainable=True)
