#!/usr/bin/env python3
"""A script that sums up i squared"""


def summation_i_squared(n):
    """A function that sums up i squared"""
    result=0
    for i in range(1, n+1):
        result+=i**2
    return result
