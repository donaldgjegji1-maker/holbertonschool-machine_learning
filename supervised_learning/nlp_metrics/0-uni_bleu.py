#!/usr/bin/env python3
"""Unigram BLEU score calculation"""
import math


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence.

    Args:
        references: list of reference translations (each a list of words)
        sentence: list of words in the proposed sentence

    Returns:
        The unigram BLEU score
    """
    # Count word occurrences in the sentence
    sentence_counts = {}
    for word in sentence:
        sentence_counts[word] = sentence_counts.get(word, 0) + 1

    # For each word, find the maximum count across all references (clipping)
    clipped_count = 0
    for word, count in sentence_counts.items():
        max_ref_count = max(
            ref.count(word) for ref in references
        )
        clipped_count += min(count, max_ref_count)

    # Precision: clipped count / sentence length
    precision = clipped_count / len(sentence)

    # Brevity penalty: pick the reference whose length is closest to sentence
    sentence_len = len(sentence)
    closest_ref_len = min(
        (abs(len(ref) - sentence_len), len(ref)) for ref in references
    )[1]

    if sentence_len >= closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_ref_len / sentence_len)

    return bp * precision
