#!/usr/bin/env python3
"""A script that slices a pd.DataFrame"""


def slice(df):
    """A function that slices a pd.DataFrame"""

    df = df[['High', 'Low', 'Close', 'Volume_(BTC)']]
    df = df.iloc[::60]
    return df
