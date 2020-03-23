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

##{
import numpy as np
import pandas as pd

# We only use pandas here to load data


# 1. Download 1-2 years of SPY. Find two other ETF that track it.
# We will get the SPY, VOO, IVV


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

# Calculate log return
spy_ret = Log_Return(spy,shift(spy,1))
voo_ret = Log_Return(voo,shift(voo,1))
ivv_ret = Log_Return(ivv,shift(ivv,1))

# Calculate active return
act_ret_voo = spy_ret - voo
act_ret_ivv = spy_ret - ivv

# Calculate average active return
act_ret_voo_ave = np.average(act_ret_voo[~np.isnan(act_ret_voo)])
act_ret_ivv_ave = np.average(act_ret_ivv[~np.isnan(act_ret_ivv)])

# Compute the tracking error and mean_adjusted_tracking_error
TE_voo = Tracking_error(len(voo), act_ret_voo[~np.isnan(act_ret_voo)], act_ret_voo_ave)
TE_ivv = Tracking_error(len(ivv), act_ret_ivv[~np.isnan(act_ret_ivv)], act_ret_ivv_ave)

MATE_voo = mean_adjusted_tracking_error(len(voo),act_ret_voo[~np.isnan(act_ret_voo)]) 
MATE_ivv = mean_adjusted_tracking_error(len(ivv),act_ret_ivv[~np.isnan(act_ret_ivv)])

## Download SPDR funds
##{

## Load data SPDR
XLB = pd.read_csv("Data/XLB.csv",
                    delimiter=',')
XLB['Date'] = pd.to_datetime(XLB['Date'])
XLB.set_index('Date', inplace=True)

XLE = pd.read_csv("Data/XLE.csv",
                    delimiter=',')
XLE['Date'] = pd.to_datetime(XLE['Date'])
XLE.set_index('Date', inplace=True)

spy = pd.read_csv("Data/SPY.csv",
                    delimiter=',')
spy['Date'] = pd.to_datetime(spy['Date'])
spy.set_index('Date', inplace=True)

spy = pd.read_csv("Data/SPY.csv",
                    delimiter=',')
spy['Date'] = pd.to_datetime(spy['Date'])
spy.set_index('Date', inplace=True)

spy = pd.read_csv("Data/SPY.csv",
                    delimiter=',')
spy['Date'] = pd.to_datetime(spy['Date'])
spy.set_index('Date', inplace=True)

spy = pd.read_csv("Data/SPY.csv",
                    delimiter=',')
spy['Date'] = pd.to_datetime(spy['Date'])
spy.set_index('Date', inplace=True)

spy = pd.read_csv("Data/SPY.csv",
                    delimiter=',')
spy['Date'] = pd.to_datetime(spy['Date'])
spy.set_index('Date', inplace=True)

spy = pd.read_csv("Data/SPY.csv",
                    delimiter=',')
spy['Date'] = pd.to_datetime(spy['Date'])
spy.set_index('Date', inplace=True)


##}

##}

