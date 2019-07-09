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

PATH = "DataSets/" # relative path to our data

n = 100 # load every nth row into df_raw

start = time.time()
df_raw = pd.read_csv(f'{PATH}train.csv', low_memory=False, skiprows=lambda i: i % n != 0)
end=time.time()
loadtingtime= end-start

