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
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(df_raw)

# distance – in units of degrees – travelled during each trip
def add_distance_feature(df):
    df['distance'] = np.sqrt(df.abs_diff_longitude**2 + df.abs_diff_latitude**2)
    
add_distance_feature(df_raw)