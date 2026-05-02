#!/usr/bin/env python3
"""Converts a gensim word2vec model to a keras Embedding layer"""
import tensorflow as tf


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a trainable keras Embedding layer"""
    keyed_vectors = model.wv
    vocab_size = len(keyed_vectors)
    vector_size = keyed_vectors.vector_size
    weights = [keyed_vectors[word] for word in keyed_vectors.index_to_key]
    return tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        embeddings_initializer=tf.keras.initializers.Constant(weights),
        trainable=True
    )
