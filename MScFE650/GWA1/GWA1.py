# Reference
# https://towardsdatascience.com/testing-for-normality-using-skewness-and-kurtosis-afd61be860

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
import scipy.stats as st

#%%
# Import data
aal = pd.read_csv("~/Projects/wqu/MScFE650/GWA1/RawData/AAL.csv", delimiter=',')
aal['Date'] = pd.to_datetime(aal['Date'])
aal['Day'] = aal['Date']
aal.set_index('Date', inplace=True)

# Plot the price
#  fig = plt.figure()
#  plt.xlabel('Day')
#  plt.ylabel('Price (USD)')
#  fig.suptitle('Stock Price for AAL')
#  #  aal_plt = plt.plot(aal['Date'], aal['AdjClose'], 'go-', label='Price AAL')
#  plt.xticks( aal['AdjClose'], aal.index.values  )
#  aal_plt = plt.plot(aal['AdjClose'], 'go-', label='Price AAL')
#  plt.legend(handles=aal_plt)
#  plt.show()

#  sns.set_style('darkgrid')
#  sns.lineplot(x='Day', y='AdjClose', data=aal)
#  plt.xticks(rotation=30)
#  plt.show()

#%%
## Calculate Return, Log Return
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
# Calculate Normal_return and log return
aal['Normal_Return'] = normal_neturn(aal['AdjClose'],shift(aal['AdjClose'],1))
aal['Log_Return'] = log_return(aal['AdjClose'],shift(aal['AdjClose'],1))

price_mean = aal['AdjClose'].mean()
price_std = aal['AdjClose'].std()

return_mean = aal['Normal_Return'].mean()
return_std = aal['Normal_Return'].std()

log_return_mean = aal['Log_Return'].mean()
log_return_std = aal['Log_Return'].std()

#%%

aal['SMA25_Price'] = aal['AdjClose'].rolling(window=25).mean()
aal['SMA25_Log_Return'] = aal['Log_Return'].rolling(window=25).mean()
aal['EWMA_25_Price'] = aal['AdjClose'].ewm(com=25).mean()

#%%
# Structural break chow_test


#%%
# Bera-Jarque
#  X = aal['Day']
#  y = aal['AdjClose']

#  olsr_results = smf.ols(y,X).fit()
#  print(olsr_results.summary())
st.jarque_bera(aal['Log_Return'])

#%%
#%%
