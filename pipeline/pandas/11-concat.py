#!/usr/bin/env python3
"""A script that takes a pd.DataFrame objects and performs operations"""

import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """A function that takes a pd.DataFrame objects and performs operations"""

    df1 = index(df1)
    df2 = index(df2)
    df2_selected = df2[df2.index <= 1417411920]
    df_concat = pd.concat([df2_selected, df1], keys=['bitstamp', 'coinbase'])
    return df_concat
