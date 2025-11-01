#!/usr/bin/env python3
"""A script that takes a pd.DataFrame as input and performs operations"""

import pandas as pd


def array(df):
    """A function that takes a pd.DataFrame as input and performs operations"""

    last10 = df[['High', 'Close']].tail(10)
    return last10.to_numpy()
