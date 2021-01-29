# import_package
import os
import pickle
from functools import reduce
from operator import mul

import pandas as pd
import numpy as np

from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn import linear_model
from sklearn.decomposition import PCA

import holoviews as hv
import hvplot
import hvplot.pandas

# check_bokeh_extension
np.random.seed(42)
hv.extension("bokeh")

# import pandas_datareader
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as pdr

apple = pd.read_csv("./Data/AAPL.csv", delimiter=",").reset_index()

dw = durbin_watson(pd.to_numeric(apple.Adj_Close).pct_change().dropna().values)
print(f"DW-statistic of {dw}")

a = 1
a

print("Trang beo")
