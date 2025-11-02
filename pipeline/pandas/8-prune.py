#!/usr/bin/env python3
"""A script that takes a pd.DataFrame and performs operations"""


def prune(df):
    """A function that takes a pd.DataFrame and performs operations"""

    return df.dropna(subset=['Close'])
