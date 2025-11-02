#!/usr/bin/env python3
"""A script that takes a pd.DataFrame objects and performs operations"""


def analyze(df):
    """A function that takes a pd.DataFrame objects and performs operations"""

    if 'Timestamp' in df.columns:
        df_numeric = df.drop(columns=['Timestamp'])
    else:
        df_numeric = df
    stats = df_numeric.describe()
    return stats
