#!/usr/bin/env python3
"""N-gram BLEU score calculation"""
import math


def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence.

    Args:
        references: list of reference translations (each a list of words)
        sentence:   list of words in the proposed sentence
        n:          size of the n-gram to use for evaluation

    Returns:
        The n-gram BLEU score
    """
    def get_ngrams(words, n):
        """Return a dict of ngram -> count for a list of words."""
        ngrams = {}
        for i in range(len(words) - n + 1):
            gram = tuple(words[i:i + n])
            ngrams[gram] = ngrams.get(gram, 0) + 1
        return ngrams

    # Build ngram counts for the sentence
    sentence_ngrams = get_ngrams(sentence, n)

    # For each ngram in the sentence, clip to max count across all references
    clipped_count = 0
    for gram, count in sentence_ngrams.items():
        max_ref_count = max(
            get_ngrams(ref, n).get(gram, 0) for ref in references
        )
        clipped_count += min(count, max_ref_count)

    # Total number of ngrams in the sentence
    total_ngrams = max(len(sentence) - n + 1, 0)

    if total_ngrams == 0:
        return 0.0

    precision = clipped_count / total_ngrams

    # Brevity penalty: closest reference length to sentence length
    sentence_len = len(sentence)
    closest_ref_len = min(
        (abs(len(ref) - sentence_len), len(ref)) for ref in references
    )[1]

    if sentence_len >= closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_ref_len / sentence_len)

    return bp * precision
