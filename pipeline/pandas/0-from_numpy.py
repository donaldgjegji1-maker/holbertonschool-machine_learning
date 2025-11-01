#!/usr/bin/env python3
"""A script that creates a pd.DataFrame from a np.ndarray"""

import pandas as pd
import numpy as np


def from_numpy(array):
    """A function that creates a pd.DataFrame from a np.ndarray"""

    num_cols = array.shape[1]
    columns = [chr(65 + i) for i in range(num_cols)]  # 65 is ASCII for 'A'
    return pd.DataFrame(array, columns=columns)
