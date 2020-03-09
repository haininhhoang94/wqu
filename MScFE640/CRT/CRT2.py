# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Collaborative Review Task 2
# 
# In this module, you are required to complete a Collaborative review task, which is designed to test your ability to apply and analyze the knowledge you have learned during the week.
# 
# ## Questions
# 
# 1. Returns
# - Download 1-2 years of price history of a stock.
# - Compute its log return.
# - Compute the mean, standard deviation, skewness, and excess kurtosis of its log return.
# - Repeat for a second stock.
# - Compute the covariance and the correlation. Explain their difference. How do you convert one to the other?
# 
# 2. Build your own transition
# - Divide the data into 2 uneven parts: the first part is 80% of your data, and the second part is 20%.
# - Categorize each day in the 1-2 year price history as belonging to one of four categories:
#     - Both stocks up
#     - Stock #1 up, stock #2 down.
#     - Stock #1 down, stock #2 up.
#     - Both stocks down
# - Build a transition matrix of portfolio direction that shows your portfolio in four scenerios:
#     - From moving together to moving together. That means starting from uu or dd & going to uu or dd.
#     - From moving together to moving apart. That means starting from uu or dd & going to ud or du.
#     - From moving apart to moving together. That means starting from ud or du & going to uu or dd.
#     - From moving apart to moving apart. That means starting from ud or du & going to ud or du.
# - How similar is the transition matrix from the first group to the second group?
# - Is the process Markovian?
# 
# ## Answers:

# %%
import numpy as np
import pandas as pd
import datetime

# %%
# Import the raw data
#tesla = np.genfromtxt('TSLA.csv',
#                      converters={0: lambda x:datetime.datetime.strptime(x.decode("utf-8"), "%m/%d/%Y")}
#                      ,delimiter=',')
#exxon = np.genfromtxt('XOM.csv',
#                      converters={0: lambda x:datetime.datetime.strptime(x.decode("utf-8"), "%m/%d/%Y")}
#                      ,delimiter=',')
tesla = pd.read_csv("TSLA.csv",
                    delimiter=',')
tesla['Date'] = pd.to_datetime(tesla['Date'])
tesla.set_index('Date', inplace=True)

exxon = pd.read_csv("XOM.csv",
                    delimiter=',')
exxon['Date'] = pd.to_datetime(exxon['Date'])
exxon.set_index('Date', inplace=True)

# %%
# Calculate daily log return
tesla["Log_Return"] = np.log(tesla['Price']).diff()
exxon["Log_Return"] = np.log(exxon['Price']).diff()

# %%
# Calculate mean, standard deviation, skewness, excess kurtosis
import scipy.stats as stats

tesla_mean = tesla['Log_Return'].mean()
telsa_std = tesla['Log_Return'].std()
tesla_skew = stats.skew(tesla['Log_Return'].dropna())
tesla_kurtosis = stats.kurtosis(tesla['Log_Return'].dropna())

exxon_mean = exxon['Log_Return'].mean()
telsa_std = exxon['Log_Return'].std()
exxon_skew = stats.skew(exxon['Log_Return'].dropna())
exxon_kurtosis = stats.kurtosis(exxon['Log_Return'].dropna())


# %%
# Calculate Covariance and Correlation
covariance_tesla_exxon = np.cov(tesla['Log_Return'].dropna(),
                                exxon['Log_Return'].dropna())
correlation_tesla_exxon = np.corrcoef(tesla['Log_Return'].dropna(),
                                      exxon['Log_Return'].dropna())

# %%
# Divide data into 2 uneven part: 80-20
from sklearn.model_selection import train_test_split
tesla_80, tesla_20 = train_test_split(tesla, test_size=0.2)
exxon_80, exxon_20 = train_test_split(exxon, test_size=0.2)

# %%
# Create a new columns decide which direction the return go based on log return
def decide_up_down(log_r):
    if (log_r >0):
        result = "1"
    else:
        result = "0"
    return result

# Vectorize:
v_decide_up_down = np.vectorize(decide_up_down)
tesla_80["UD"] = v_decide_up_down(tesla_80["Log_Return"])
tesla_20["UD"] = v_decide_up_down(tesla_20["Log_Return"])
exxon_80["UD"] = v_decide_up_down(exxon_80["Log_Return"])
exxon_20["UD"] = v_decide_up_down(exxon_20["Log_Return"])


# %%
# We create portfolio consist of 2 data set: 80 and 20
def create_portfolio_ud(tesla, exxon):
    tesla["EX_UD"] = exxon["UD"]
    tesla.rename(columns={"UD": "TL_UD"}, inplace=True)
    portfolio = tesla[["TL_UD",'EX_UD']]
    portfolio.dropna(inplace=True)
    return portfolio

# %%
portfolio_80 = create_portfolio_ud(tesla_80,exxon_80)
portfolio_20 = create_portfolio_ud(tesla_20,exxon_20)

# %%
# A is apart (status change), T is together (status unchanged)

# %%
# Build matrix: overall 80
#mt_mt = (overall_80['UU'] + overall_80['DD']) / overall_80.sum()

# %%
