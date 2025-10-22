#!/usr/bin/env python3
"""A script that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """A function that calculates the derivative of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    derivative = []
    for power in range(1, len(poly)):
        derivative.append(power * poly[power])
    if all(coeff == 0 for coeff in derivative):
        return [0]
    return derivative
