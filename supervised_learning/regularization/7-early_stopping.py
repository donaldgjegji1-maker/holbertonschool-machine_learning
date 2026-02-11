#!/usr/bin/env python3
"""
Early Stopping Module
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should be stopped early.
    """
    if opt_cost - cost > threshold:
        # Improvement detected, reset count
        count = 0
    else:
        count += 1

    should_stop = count >= patience
    return should_stop, count
