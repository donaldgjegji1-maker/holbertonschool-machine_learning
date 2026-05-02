#!/usr/bin/env python3
"""
Module to create a Bag of Words embedding matrix
"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

    Args:
        sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis
               If None, all words within sentences should be used

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: list of the features used for embeddings
    """
    # Pre-process sentences: lowercase and remove punctuation/possessives
    # The regex [^a-zA-Z ] helps match the specific tokenization
    processed_sentences = []
    for s in sentences:
        # Standardize "children's" to "children" and remove punctuation
        s = s.lower()
        s = re.sub(r"'s\b", "", s)
        s = re.sub(r"[^a-z\s]", " ", s)
        processed_sentences.append(s.split())

    if vocab is None:
        # Build vocabulary from all words found in sentences
        all_words = set()
        for sentence in processed_sentences:
            for word in sentence:
                all_words.add(word)
        features = sorted(list(all_words))
    else:
        features = vocab

    # Map words to their index for faster lookup
    word_to_idx = {word: i for i, word in enumerate(features)}

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, sentence in enumerate(processed_sentences):
        for word in sentence:
            if word in word_to_idx:
                embeddings[i, word_to_idx[word]] += 1

    return embeddings, features
