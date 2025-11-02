#!/usr/bin/env python3
"""A script that takes a pd.DataFrame and performs operations"""


def fill(df):
    """A function that takes a pd.DataFrame and performs operations"""

    df = df.drop(columns=['Weighted_Price'])
    df['Close'] = df['Close'].fillna(method='ffill')
    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    return df
