#!/usr/bin/env python3
"""
Moving Average Module
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set with bias correction.
    """
    moving_averages = []
    v = 0

    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]

        v_corrected = v / (1 - beta ** (i + 1))
        moving_averages.append(v_corrected)

    return moving_averages
