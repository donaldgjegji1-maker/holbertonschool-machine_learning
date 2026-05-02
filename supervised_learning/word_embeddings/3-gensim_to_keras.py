#!/usr/bin/env python3
"""
Module to convert a gensim word2vec model to a keras Embedding layer
"""
import tensorflow.keras as keras


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer

    Args:
        model: a trained gensim word2vec model

    Returns:
        A trainable keras Embedding layer initialized with gensim weights
    """
    # Extract the keyed vectors from the gensim model
    keyed_vectors = model.wv

    # Get the weights (the matrix of word embeddings)
    weights = keyed_vectors.vectors

    # vocab_size is the number of words, vector_size is the embedding dimension
    vocab_size, vector_size = weights.shape

    # Create the Keras Embedding layer
    # input_dim: size of the vocabulary
    # output_dim: size of the dense vector
    # weights: initialized with our gensim weights
    # trainable: set to True per requirements
    embedding_layer = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        embeddings_initializer=keras.initializers.Constant(weights),
        trainable=True
    )

    return embedding_layer
