#!/usr/bin/env python3
"""
Module to create a TF-IDF embedding matrix
"""
import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix
    """
    # 1. Pre-process sentences
    processed = []
    for s in sentences:
        s = s.lower()
        s = re.sub(r"'s\b", "", s)
        s = re.sub(r"[^a-z\s]", " ", s)
        processed.append(s.split())

    # 2. Determine the vocabulary
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
    # Using Relative TF: (count of word in doc) / (total words in doc)
    tf = np.zeros((s, f))
    for i, sentence in enumerate(processed):
        for word in sentence:
            if word in word_to_idx:
                tf[i, word_to_idx[word]] += 1
        # Divide by total words in the sentence (if sentence not empty)
        if len(sentence) > 0:
            tf[i] /= len(sentence)

    # 4. Calculate Inverse Document Frequency (IDF)
    # df is the number of documents containing the word
    df = np.count_nonzero(tf, axis=0)

    # Calculate IDF: log(N / df)
    # Using np.log (natural log) is standard for these tasks
    with np.errstate(divide='ignore'):
        idf = np.log(s / df)
    idf[np.isinf(idf)] = 0

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

    return embeddings, np.array(features)
