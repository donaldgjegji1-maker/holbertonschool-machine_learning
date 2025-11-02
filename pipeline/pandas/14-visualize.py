#!/usr/bin/env python3
"""A script to visualize the pd.DataFrame"""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

#Load the CSV
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

#Remove Weighted_Price column if it exists
if 'Weighted_Price' in df.columns:
    df = df.drop(columns=['Weighted_Price'])

#Rename Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

#Convert timestamp to datetime
df['Date'] = pd.to_datetime(df['Date'], unit='s')

#Set Date as index
df = df.set_index('Date')

#Fill missing values in Close with previous row
df['Close'] = df['Close'].ffill()

#Fill missing values in High, Low, Open with same row's Close
for col in ['High', 'Low', 'Open']:
    df[col] = df[col].fillna(df['Close'])

#Fill missing values in Volume columns with 0
for col in ['Volume_(BTC)', 'Volume_(Currency)']:
    df[col] = df[col].fillna(0)

#Filter for 2017 and beyond
df = df[df.index >= '2017-01-01']

#Resample to daily intervals and aggregate
agg_dict = {
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
}
daily_df = df.resample('D').agg(agg_dict)

#Plot the data
daily_df.plot(figsize=(15, 7), title='Daily Cryptocurrency Data from 2017')
plt.show()

#Return the transformed DataFrame
daily_df
