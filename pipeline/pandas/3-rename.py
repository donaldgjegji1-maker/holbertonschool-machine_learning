#!/usr/bin/env python3
"""A script that takes a pd.DataFrame as input and performs operations"""

import pandas as pd


def rename(df):
    """A function that takes a pd.DataFrame as input and performs operations"""

    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df[['Datetime', 'Close']]
    return df
