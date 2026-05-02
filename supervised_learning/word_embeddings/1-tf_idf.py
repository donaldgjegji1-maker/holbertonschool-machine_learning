#!/usr/bin/env python3
"""
Module to create TF-IDF embeddings for a list of sentences
"""
import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """
    Creates TF-IDF embeddings for a list of sentences.

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use for analysis
               If None, all words within sentences should be used

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: list of the features used for embeddings
    """
    # Preprocess sentences: convert to lowercase and extract words
    def preprocess(text):
        text = text.lower()
        # Split on non-alphanumeric characters
        words = re.findall(r'[a-z0-9]+', text)
        return words

    # Process all sentences
    processed_sentences = [preprocess(sentence) for sentence in sentences]

    # Determine vocabulary if not provided
    if vocab is None:
        # Collect all unique words
        unique_words = set()
        for sentence_words in processed_sentences:
            unique_words.update(sentence_words)
        vocab = sorted(list(unique_words))

    s = len(sentences)
    f = len(vocab)

    # Create term frequency matrix (using raw counts)
    tf_matrix = np.zeros((s, f))
    for i, sentence_words in enumerate(processed_sentences):
        for word in sentence_words:
            if word in vocab:
                j = vocab.index(word)
                tf_matrix[i, j] += 1

    # Calculate document frequency
    df = np.zeros(f)
    for j, word in enumerate(vocab):
        for sentence_words in processed_sentences:
            if word in sentence_words:
                df[j] += 1

    # Calculate IDF
    idf = np.zeros(f)
    for j in range(f):
        if df[j] > 0:
            idf[j] = np.log(s / df[j])
        else:
            idf[j] = 0

    # Calculate TF-IDF
    embeddings = tf_matrix * idf

    # L2 normalize each row
    for i in range(s):
        norm = np.linalg.norm(embeddings[i])
        if norm > 0:
            embeddings[i] = embeddings[i] / norm

    return embeddings, vocab
