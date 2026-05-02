#!/usr/bin/env python3
"""
Module to create, build and train a FastText model using gensim
"""
from gensim.models import FastText


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds and trains a gensim fastText model

    Args:
        sentences: list of sentences to be trained on
        vector_size: dimensionality of the embedding layer
        min_count: minimum occurrences of a word for use in training
        negative: size of negative sampling
        window: maximum distance between current and predicted word
        cbow: boolean (True for CBOW, False for Skip-gram)
        epochs: number of iterations to train over
        seed: seed for the random number generator
        workers: number of worker threads to train the model

    Returns:
        The trained fastText model
    """
    # Set the training algorithm: 0 for CBOW, 1 for Skip-gram
    training_alg = 0 if cbow else 1

    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=training_alg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
