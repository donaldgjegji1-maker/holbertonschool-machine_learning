#!/usr/bin/env python3
"""A script that takes a pd.DataFrame and performs operations"""


def flip_switch(df):
    """A function that takes a pd.DataFrame and performs operations"""

    df = df.sort_index(ascending=False)
    df = df.transpose()
    return df
