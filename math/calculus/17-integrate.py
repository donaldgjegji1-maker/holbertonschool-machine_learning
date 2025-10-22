#!/usr/bin/env python3
"""A script that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """A function that calculates the integral of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, (int, float)):
        return None
    integral = [C]
    for power, coeff in enumerate(poly):
        new_coeff = coeff / (power + 1)
        if new_coeff.is_integer():
            integral.append(int(new_coeff))
        else:
            integral.append(new_coeff)
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    return integral
