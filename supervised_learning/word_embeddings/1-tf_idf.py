#!/usr/bin/env python3
"""
Module to create a TF-IDF embedding matrix
"""
import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix

    Args:
        sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: list of the features used for embeddings
    """
    # 1. Pre-process sentences to match the BoW logic
    processed = []
    for s in sentences:
        # Standardize "children's" to "children" and remove punctuation
        s = s.lower()
        s = re.sub(r"'s\b", "", s)
        s = re.sub(r"[^a-z\s]", " ", s)
        processed.append(s.split())

    # 2. Determine the vocabulary (features)
    if vocab is None:
        all_words = set()
        for sentence in processed:
            for word in sentence:
                all_words.add(word)
        features = sorted(list(all_words))
    else:
        features = vocab

    word_to_idx = {word: i for i, word in enumerate(features)}
    s = len(sentences)
    f = len(features)

    # 3. Calculate Term Frequency (TF)
    tf = np.zeros((s, f))
    for i, sentence in enumerate(processed):
        for word in sentence:
            if word in word_to_idx:
                tf[i, word_to_idx[word]] += 1

    # 4. Calculate Inverse Document Frequency (IDF)
    # Using the natural log formula: ln(total sentences / sentences with word)
    df = np.count_nonzero(tf, axis=0)

    # Avoid division by zero for words in vocab that don't appear in sentences
    with np.errstate(divide='ignore'):
        idf = np.log(s / df)
    idf[np.isinf(idf)] = 0  # Set IDF to 0 for words not found in any sentence

    # 5. Calculate TF-IDF
    embeddings = tf * idf

    # 6. L2 Normalization
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.divide(
        embeddings,
        norm,
        out=np.zeros_like(embeddings),
        where=norm != 0
    )

    return embeddings, features
