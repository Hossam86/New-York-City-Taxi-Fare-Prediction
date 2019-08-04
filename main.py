import math
import os
import re
import time
from datetime import datetime

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas_summary import DataFrameSummary
from sklearn import metrics
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              forest)
from sklearn.tree import export_graphviz

PATH = "resources/new-york-city-taxi-fare-prediction/"  # relative path to our data

n = 100  # load every nth row into df_raw

start = time.time()
df_raw = pd.read_csv(f'{PATH}train.csv', low_memory=False,
                     skiprows=lambda i: i % n != 0)
end = time.time()
loadtingtime = end-start

# this gives us about half a million rows, plent for quick prototyping
print(df_raw.shape)
print(df_raw.head())
print(df_raw.info())
print(df_raw.describe())

print(f'Old size: {len(df_raw)}')

min_fare = 0
max_pass = 10
lat_range = [30, 50]
lon_range = [-85, -65]

df_raw = df_raw[(df_raw.pickup_latitude > lat_range[0]) &
                (df_raw.pickup_latitude < lat_range[1]) &
                (df_raw.pickup_longitude > lon_range[0]) &
                (df_raw.pickup_longitude < lon_range[1]) &
                (df_raw.dropoff_latitude > lat_range[0]) &
                (df_raw.dropoff_latitude < lat_range[1]) &
                (df_raw.dropoff_longitude > lon_range[0]) &
                (df_raw.dropoff_longitude < lon_range[1]) &
                (df_raw.fare_amount > min_fare) &
                (df_raw.passenger_count < max_pass)]
print(f'New size: {len(df_raw)}')

'''Pre-processing Distance feature engineering '''
# =================================================

# create two new features representing the latitude and longitude vectors traversed during the trip


def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (
        df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(df_raw)

# distance – in units of degrees – travelled during each trip


def add_distance_feature(df):
    df['distance'] = np.sqrt(df.abs_diff_longitude **
                             2 + df.abs_diff_latitude**2)


add_distance_feature(df_raw)

plot = df_raw.iloc[:2000].plot.scatter(
    'abs_diff_longitude', 'abs_diff_latitude', alpha=0.5, s=7)

plt.show()

''' Let's take a quick look at the distribution for both distance travelled and taxi fare within a limited range.
 These limits will help us better see the shape of the distribution.'''
short_rides = df_raw[(df_raw['abs_diff_latitude'] < 0.1)
                     & (df_raw['abs_diff_longitude'] < 0.1)]

# Kernel Density Plot for distance travelled
fig = plt.figure(figsize=(15, 4),)
ax = sns.kdeplot(short_rides.distance, color='steelblue',
                 shade=True, label='distance')
plt.title('Taxi Ride Distance Distribution')
plt.show()

'''The distance travelled follows a positive skewed normal distribution, with the max frequency at less than 0.02 degrees (about 1.4 miles).
 This makes intuitive sense. We'd expect most rides in NYC to be short, likely within Manhattan, with a few longer rides.'''

cheap_rides = df_raw[df_raw.fare_amount < 50]

# Kernel Density Plot for taxi fare
fig = plt.figure(figsize=(15,4),)
ax=sns.kdeplot(cheap_rides.fare_amount , color='red',shade=True,label='fare_amount')
plt.title('Taxi Ride Fare')
plt.show()

print("Correlation between taxi fare and distance travelled: " +
      f"{df_raw['fare_amount'].corr(df_raw['distance'])}")

# Random Forests
# Quick Baseline Model
# =======================================================================
X_train = df_raw.drop(['key', 'pickup_datetime', 'fare_amount'], axis=1)
y_train = df_raw.fare_amount


def split_vals(df, n): return df[:n].copy(), df[n:].copy()

n_valid = 9914  # same as Kaggle's test set size
n_trn = len(X_train)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(X_train, n_trn)
y_train, y_valid = split_vals(y_train, n_trn)

m = RandomForestRegressor()
m.fit(X_train, y_train)
m.score(X_train, y_train)

def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)