#!/usr/bin/env python3
"""A script that takes a pd.DataFrame and performs operations"""


def high(df):
    """A script that takes a pd.DataFrame and performs operations"""

    return df.sort_values('High', ascending=False)
