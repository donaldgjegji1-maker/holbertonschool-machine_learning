#!/usr/bin/env python3
"""A script that concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """A script that concatenates two matrices along a specific axis"""

    if axis == 0:
        if len(mat1) > 0 and len(mat2) > 0 and len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    elif axis == 1:
        if len(mat1) != len(mat2):
            return None

        result = []
        for i in range(len(mat1)):
            if len(mat1[i]) != len(mat1[0]) or len(mat2[i]) != len(mat2[0]):
                return None
            result.append(mat1[i][:] + mat2[i][:])
        return result

    else:
        return None
