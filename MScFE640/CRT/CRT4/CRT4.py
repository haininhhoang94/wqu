#%%
# Collaborative Review Task 4
# In this module, you are required to complete a collaborative review task, which is designed to test your ability to apply and analayze
# the knowledge you have learned during the week

# Question:
# 1. Download 1-2 years of SPY. Find two other ETF that track it.
# 2. Compute the return, active returns, and average active return.
# 3. Compute the tracking error and mean-adjusted tracking error.
# 4. Which ETF tracks the S&P500 better?
# 5. Download the Select SPDR funds (tickers = XLB, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY) over the same time period as you did
# for SPY.
# 6. Compute the returns.
# 7. Write a function that computes active return internally and uses that to compute the mean-adjusted tracking error.
# 8. Determine which single sector fund best tracks the S&P500.

#%%
import numpy as np
import pandas as pd

# We only use pandas here to load data


# 1. Download 1-2 years of SPY. Find two other ETF that track it.
# We will get the SPY, VOO, IVV

#%%
spy = pd.read_csv("Data/SPY.csv",
                    delimiter=',')
spy['Date'] = pd.to_datetime(spy['Date'])
spy.set_index('Date', inplace=True)

ivv = pd.read_csv("Data/IVV.csv",
                    delimiter=',')
ivv['Date'] = pd.to_datetime(ivv['Date'])
ivv.set_index('Date', inplace=True)

voo = pd.read_csv("Data/VOO.csv",
                    delimiter=',')
voo['Date'] = pd.to_datetime(voo['Date'])
voo.set_index('Date', inplace=True)

# Convert from pandas to numpy
spy = np.array(spy['Adj Close'])
ivv = np.array(ivv['Adj Close'])
voo = np.array(voo['Adj Close'])

#%%
# 2. Compute the return, active returns, and average active return.

# Write function to compute return, active return
# Note that we will use log return here
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

def Normal_Return(P_f, P_i):
    result = (P_f - P_i)/P_i
    return result

def Log_Return(P_f, P_i):
    result = np.log(P_f/P_i)
    return result

# Tracking errors is the standard deviation of active return
def Tracking_error(T, R_i, R_ave):
    a = 1/(T-1)
    b = np.sum((R_i - R_ave)**2)
    result = (a*b)**(1/2)
    return result

def mean_adjusted_tracking_error(T, active_return):
    a = 1/(T-1)
    b = np.sum(active_return**2)
    result = (a*b)**(1/2)
    return result

#%%
# Calculate log return
spy_ret = Log_Return(spy,shift(spy,1))
voo_ret = Log_Return(voo,shift(voo,1))
ivv_ret = Log_Return(ivv,shift(ivv,1))

# Calculate active return
act_ret_voo = voo_ret - spy_ret
act_ret_ivv = ivv_ret - spy_ret

# Calculate average active return
act_ret_voo_ave = np.average(act_ret_voo[~np.isnan(act_ret_voo)])
act_ret_ivv_ave = np.average(act_ret_ivv[~np.isnan(act_ret_ivv)])

# Compute the tracking error and mean_adjusted_tracking_error
TE_voo = Tracking_error(len(voo), act_ret_voo[~np.isnan(act_ret_voo)], act_ret_voo_ave)
TE_ivv = Tracking_error(len(ivv), act_ret_ivv[~np.isnan(act_ret_ivv)], act_ret_ivv_ave)

MATE_voo = mean_adjusted_tracking_error(len(voo),act_ret_voo[~np.isnan(act_ret_voo)]) 
MATE_ivv = mean_adjusted_tracking_error(len(ivv),act_ret_ivv[~np.isnan(act_ret_ivv)])


#%%
# Download SPDR funds
# Load data SPDR
XLB = pd.read_csv("Data/XLB.csv",
                    delimiter=',')
XLB['Date'] = pd.to_datetime(XLB['Date'])
XLB.set_index('Date', inplace=True)

XLE = pd.read_csv("Data/XLE.csv",
                    delimiter=',')
XLE['Date'] = pd.to_datetime(XLE['Date'])
XLE.set_index('Date', inplace=True)

XLF = pd.read_csv("Data/XLF.csv",
                    delimiter=',')
XLF['Date'] = pd.to_datetime(XLF['Date'])
XLF.set_index('Date', inplace=True)

XLP = pd.read_csv("Data/XLP.csv",
                    delimiter=',')
XLP['Date'] = pd.to_datetime(XLP['Date'])
XLP.set_index('Date', inplace=True)

XLK = pd.read_csv("Data/XLK.csv",
                    delimiter=',')
XLK['Date'] = pd.to_datetime(XLK['Date'])
XLK.set_index('Date', inplace=True)

XLRE = pd.read_csv("Data/XLRE.csv",
                    delimiter=',')
XLRE['Date'] = pd.to_datetime(XLRE['Date'])
XLRE.set_index('Date', inplace=True)

XLU = pd.read_csv("Data/XLU.csv",
                    delimiter=',')
XLU['Date'] = pd.to_datetime(XLU['Date'])
XLU.set_index('Date', inplace=True)

XLV = pd.read_csv("Data/XLV.csv",
                    delimiter=',')
XLV['Date'] = pd.to_datetime(XLV['Date'])
XLV.set_index('Date', inplace=True)

XLY = pd.read_csv("Data/XLY.csv",
                    delimiter=',')
XLY['Date'] = pd.to_datetime(XLY['Date'])
XLY.set_index('Date', inplace=True)

SP500 = pd.read_csv("Data/GSPC.csv",
                    delimiter=',')
SP500['Date'] = pd.to_datetime(SP500['Date'])
SP500.set_index('Date', inplace=True)

# Covert pandas dataframe to numpy
XLB = np.array(XLB['Adj Close'])
XLE = np.array(XLE['Adj Close'])
XLF = np.array(XLF['Adj Close'])
XLK = np.array(XLK['Adj Close'])
XLP = np.array(XLP['Adj Close'])
XLRE = np.array(XLRE['Adj Close'])
XLU = np.array(XLU['Adj Close'])
XLV = np.array(XLV['Adj Close'])
XLY = np.array(XLY['Adj Close'])
SP500 = np.array(SP500['Adj Close'])

#%%
# Calculate log return for each
XLB_ret = Log_Return(XLB,shift(XLB,1))
XLE_ret = Log_Return(XLE,shift(XLE,1))
XLF_ret = Log_Return(XLF,shift(XLF,1))
XLK_ret = Log_Return(XLK,shift(XLK,1))
XLP_ret = Log_Return(XLP,shift(XLP,1))
XLRE_ret = Log_Return(XLRE,shift(XLRE,1))
XLU_ret = Log_Return(XLU,shift(XLU,1))
XLV_ret = Log_Return(XLV,shift(XLV,1))
XLY_ret = Log_Return(XLY,shift(XLY,1))


#%%
# Write a function that computes active return internally and uses that
# to compute the mean_adjusted_tracking_error
def MATE_alter(T,R_p, R_b):
    a = 1/T
    active_return = R_p - R_b
    b = np.sum(active_return**2)
    result = (a*b)**(1/2)
    return result
# Compute MATE 
MATE_XLB = MATE_alter(len(XLB),XLB_ret[~np.isnan(XLB_ret)], spy_ret[~np.isnan(spy_ret)]) 
MATE_XLE = MATE_alter(len(XLE),XLE_ret[~np.isnan(XLE_ret)], spy_ret[~np.isnan(spy_ret)]) 
MATE_XLF = MATE_alter(len(XLF),XLF_ret[~np.isnan(XLF_ret)], spy_ret[~np.isnan(spy_ret)]) 
MATE_XLK = MATE_alter(len(XLK),XLK_ret[~np.isnan(XLK_ret)], spy_ret[~np.isnan(spy_ret)]) 
MATE_XLP = MATE_alter(len(XLP),XLP_ret[~np.isnan(XLP_ret)], spy_ret[~np.isnan(spy_ret)]) 
MATE_XLRE = MATE_alter(len(XLRE),XLRE_ret[~np.isnan(XLRE_ret)], spy_ret[~np.isnan(spy_ret)]) 
MATE_XLU = MATE_alter(len(XLU),XLU_ret[~np.isnan(XLU_ret)], spy_ret[~np.isnan(spy_ret)]) 
MATE_XLV = MATE_alter(len(XLV),XLV_ret[~np.isnan(XLV_ret)], spy_ret[~np.isnan(spy_ret)]) 
MATE_XLY = MATE_alter(len(XLY),XLY_ret[~np.isnan(XLY_ret)], spy_ret[~np.isnan(spy_ret)]) 






# %%
