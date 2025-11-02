#!/usr/bin/env python3
"""A script that takes a pd.DataFrame objects and performs operations"""

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """A function that takes a pd.DataFrame objects and performs operations"""

    df1 = index(df1)
    df2 = index(df2)
    df1_sel = df1[(df1.index >= 1417411980) & (df1.index <= 1417417980)]
    df2_sel = df2[(df2.index >= 1417411980) & (df2.index <= 1417417980)]
    df_concat = pd.concat([df2_sel, df1_sel], keys=['bitstamp', 'coinbase'])
    df_concat = df_concat.reorder_levels([1, 0])
    df_concat = df_concat.sort_index(level=0)

    return df_concat
