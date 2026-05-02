#!/usr/bin/env python3
"""
Module to create a Bag of Words embedding matrix
"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix
    """
    processed_sentences = []
    for s in sentences:
        s = s.lower()
        # Handle possessives specifically for the "children's" case
        s = re.sub(r"'s\b", "", s)
        # Replace non-alphabetic characters with spaces
        s = re.sub(r"[^a-z\s]", " ", s)
        processed_sentences.append(s.split())

    if vocab is None:
        all_words = set()
        for sentence in processed_sentences:
            for word in sentence:
                all_words.add(word)
        features = sorted(list(all_words))
    else:
        features = vocab

    word_to_idx = {word: i for i, word in enumerate(features)}

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, sentence in enumerate(processed_sentences):
        for word in sentence:
            if word in word_to_idx:
                embeddings[i, word_to_idx[word]] += 1

    # Cast features to a numpy array
    return embeddings, np.array(features)
