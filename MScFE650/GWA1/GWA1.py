# Reference
# https://towardsdatascience.com/testing-for-normality-using-skewness-and-kurtosis-afd61be860

#%%
import sys
sys.path.append('/home/haininhhoang94/Projects/wqu/MScFE650/GWA1/')
#%%
import chow_test

#%%
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import datetime
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.api as sm
import seaborn as sns
from scipy import stats

#%%
# Import data
aal = pd.read_csv("~/Projects/wqu/MScFE650/GWA1/RawData/AAL.csv", delimiter=',')
aal['Date'] = pd.to_datetime(aal['Date'], format="%d/%m/%Y")
aal['Date_'] = aal['Date']
#  aal.set_index('Date', inplace=True)

#%%
sns.set_style('darkgrid')
sns.lineplot(x='Date_', y='AdjClose', data=aal)
plt.xticks(rotation=30)
plt.show()
#  aal['AdjClose'].plot(figsize=(16/12))

#%%
# Calculate Return, Log Return
def nans(shape, dtype=float):
    # To generate nans array
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

def shift(price_array, n):
    # n is the number shift to the right
    result = nans(len(price_array))
    for i in range(n,len(price_array)):
        result[i] = price_array[i-n]
    return result

def normal_neturn(P_f, P_i):
    result = (P_f - P_i)/P_i
    return result

def log_return(P_f, P_i):
    result = np.log(P_f/P_i)
    return result

#%%
# Calculate mean and standard deviation
aal['Normal_Return'] = normal_neturn(aal['AdjClose'],shift(aal['AdjClose'],1))
aal['Log_Return'] = log_return(aal['AdjClose'],shift(aal['AdjClose'],1))

price_mean = aal['AdjClose'].mean()
price_std = aal['AdjClose'].std()

return_mean = aal['Normal_Return'].mean()
return_std = aal['Normal_Return'].std()

log_return_mean = aal['Log_Return'].mean()
log_return_std = aal['Log_Return'].std()

#%%
sns.set_style('darkgrid')
sns.lineplot(x='Date_', y='Normal_Return', data=aal)
plt.xticks(rotation=30)
plt.show()

#%%
sns.set_style('darkgrid')
sns.lineplot(x='Date_', y='Log_Return', data=aal)
plt.xticks(rotation=30)
plt.show()


#%%
# Calculate SMA(25) ad EWMA_25
aal['SMA25_Price'] = aal['AdjClose'].rolling(window=25).mean()
aal['SMA25_Log_Return'] = aal['Log_Return'].rolling(window=25).mean()
aal['EWMA_25_Price'] = aal['AdjClose'].ewm(com=25).mean()

#%%
# Structural break chow_test
# https://medium.com/@remycanario17/the-chow-test-dealing-with-heterogeneity-in-python-1b9057f0f07a
# Look at our graph, we can easily see that there is a structure break in Feb
# 2020. So we will test linear regression with y1, y2 break from 1 Feb 2020
# because of corona virus
# Note that we won't use linear regression in our return

# Case 1: with breakdown in 1-Feb-2020
y1 = aal[aal['Date_'] < '2020-02-01']['AdjClose']
x1 = np.array(aal[aal['Date_'] < '2020-02-01'].index)
y2 = aal[aal['Date_'] >='2020-02-01']['AdjClose']
x2 = np.array(aal[aal['Date_'] >= '2020-02-01'].index)

f_test = chow_test.f_value(y1, x1, y2, x2)
p_val = chow_test.p_value(y1, x1, y2, x2)
print(f_test)
print(p_val)
# => p-value = 9.681119063302348e-18 < 0.0005
# Structural break at Feb 2020 
# fit with 12/13 Feb 2020? Korean outbreak
#%%

# Case 2: with breakdown in 13-Feb-2020
y1 = aal[aal['Date_'] < '2020-02-13']['AdjClose']
x1 = np.array(aal[aal['Date_'] < '2020-02-13'].index)
y2 = aal[aal['Date_'] >='2020-02-13']['AdjClose']
x2 = np.array(aal[aal['Date_'] >= '2020-02-13'].index)

f_test = chow_test.f_value(y1, x1, y2, x2)
p_val = chow_test.p_value(y1, x1, y2, x2)
print(f_test)
print(p_val)

#%%
# Bera-Jarque
stats.jarque_bera(np.array(aal['AdjClose']))
#%%
from scipy import stats
import numpy as np
np.random.seed(987654321)
x = np.random.normal(0, 1, 100000)
y = np.random.rayleigh(1, 100000)
#%%
stats.jarque_bera(x)
#%%
plt.figure(figsize=(12,7))
ax = sns.distplot(aal['Log_Return'])
plt.show()
#%%
#%%
#%%
#%%
#%%
#%%
