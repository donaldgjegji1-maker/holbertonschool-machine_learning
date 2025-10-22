#!/usr/bin/env python3
"""A script that sums up i squared"""


def summation_i_squared(n):
    """A function that sums up i squared"""
    if not isinstance(n, int) or n < 1:
        return None
    return n**3/3 + n**2/2 + n/6
