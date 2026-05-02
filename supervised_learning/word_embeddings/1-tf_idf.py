#!/usr/bin/env python3
"""TF-IDF embedding"""
import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """Creates a TF-IDF embedding"""
    # Normalize sentences: lowercase, strip punctuation except apostrophes
    def normalize(sentence):
        sentence = sentence.lower()
        # keep apostrophes attached (children's -> children's)
        sentence = re.sub(r"[^\w\s']", '', sentence)
        return sentence.split()

    tokenized = [normalize(s) for s in sentences]

    # Flatten tokens and strip trailing apostrophes/possessives
    def clean_token(token):
        # turn "children's" -> "children"
        token = re.sub(r"'s$", '', token)
        return token.strip("'")

    cleaned = [[clean_token(t) for t in tokens] for tokens in tokenized]

    # Build vocab from all sentences if not provided
    if vocab is None:
        seen = []
        for tokens in cleaned:
            for t in tokens:
                if t not in seen:
                    seen.append(t)
        features = sorted(seen)
    else:
        features = list(vocab)

    s = len(sentences)
    f = len(features)
    feat_index = {feat: i for i, feat in enumerate(features)}

    # TF: term frequency per sentence (raw count / total words in sentence)
    tf = np.zeros((s, f))
    for i, tokens in enumerate(cleaned):
        total = len(tokens)
        if total == 0:
            continue
        for token in tokens:
            if token in feat_index:
                tf[i][feat_index[token]] += 1
        tf[i] /= total

    # IDF: log((1 + s) / (1 + df)) + 1  — sklearn smooth variant
    idf = np.zeros(f)
    for j, feat in enu
