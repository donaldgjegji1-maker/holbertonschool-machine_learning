#!/usr/bin/env python3
"""A script that returns a matrix transposed"""


def matrix_transpose(matrix):
    """
    Return the transpose of a matrix.
    Rows become columns and columns become rows.
    """

    rows = len(matrix)
    cols = len(matrix[0])

    # Create result matrix with swapped dimensions
    result = [[0 for _ in range(rows)] for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]
    return result
