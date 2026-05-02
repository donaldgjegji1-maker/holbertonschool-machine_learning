#!/usr/bin/env python3
"""TF-IDF embedding"""
import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """Creates a TF-IDF embedding"""

    def normalize(sentence):
        """Lowercase and strip punctuation, keep apostrophes"""
        sentence = sentence.lower()
        sentence = re.sub(r"[^\w\s']", '', sentence)
        return sentence.split()

    def clean_token(token):
        """Strip possessives and stray apostrophes"""
        token = re.sub(r"'s$", '', token)
        return token.strip("'")

    tokenized = [[clean_token(t) for t in normalize(s)] for s in sentences]

    if vocab is None:
        seen = []
        for tokens in tokenized:
            for t in tokens:
                if t not in seen:
                    seen.append(t)
        features = np.array(sorted(seen))
    else:
        features = np.array(list(vocab))

    s = len(sentences)
    f = len(features)
    feat_index = {feat: i for i, feat in enumerate(features)}

    # TF
    tf = np.zeros((s, f))
    for i, tokens in enumerate(tokenized):
        total = len(tokens)
        if total == 0:
            continue
        for token in tokens:
            if token in feat_index:
                tf[i][feat_index[token]] += 1
        tf[i] /= total

    # IDF (sklearn smooth variant)
    idf = np.zeros(f)
    for j, feat in enumerate(features):
        df = sum(1 for tokens in tokenized if feat in tokens)
        idf[j] = np.log((1 + s) / (1 + df)) + 1

    # TF-IDF
    tfidf = tf * idf

    # L2 normalise rows
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = tfidf / norms

    return embeddings, features
