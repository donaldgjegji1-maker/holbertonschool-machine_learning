#!/usr/bin/env python3
"""Cumulative N-gram BLEU score calculation"""
import math


def cumulative_bleu(references, sentence, n):
    """Calculates the cumulative n-gram BLEU score for a sentence.

    Args:
        references: list of reference translations (each a list of words)
        sentence:   list of words in the proposed sentence
        n:          size of the largest n-gram to use for evaluation

    Returns:
        The cumulative n-gram BLEU score
    """
    def get_ngrams(words, n):
        """Return a dict of ngram -> count for a list of words."""
        ngrams = {}
        for i in range(len(words) - n + 1):
            gram = tuple(words[i:i + n])
            ngrams[gram] = ngrams.get(gram, 0) + 1
        return ngrams

    def clipped_precision(order):
        """Compute clipped precision for a given n-gram order."""
        sentence_ngrams = get_ngrams(sentence, order)
        total = max(len(sentence) - order + 1, 0)
        if total == 0:
            return 0.0
        clipped = 0
        for gram, count in sentence_ngrams.items():
            max_ref = max(
                get_ngrams(ref, order).get(gram, 0) for ref in references
            )
            clipped += min(count, max_ref)
        return clipped / total

    # Uniform weight for each n-gram order
    weight = 1 / n

    # Weighted log sum of precisions (geometric mean)
    log_sum = 0.0
    for order in range(1, n + 1):
        p = clipped_precision(order)
        if p == 0:
            log_sum = float('-inf')
            break
        log_sum += weight * math.log(p)

    # Brevity penalty
    sentence_len = len(sentence)
    closest_ref_len = min(
        (abs(len(ref) - sentence_len), len(ref)) for ref in references
    )[1]

    if sentence_len >= closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_ref_len / sentence_len)

    return bp * math.exp(log_sum)
