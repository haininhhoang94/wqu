##{
import numpy as np
# Jorge Israel Peña - StackOverFlow
# For creating empty np array
def nans(shape, dtype=float):
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

# Example from Carol Alexander's (2008) book
fund = np.array([100,91,104,127,167,190,206,234,260,271,346,256,221,223,243,262,273])
bench1 = np.array([100,90,104,124,161,186,204,235,258,271,339,254,216,216,238,262,275])
bench2 = np.array([100,93,110,136,182,216,245,291,330,360,460,355,311,321,364,413,447])

b1_ret = Normal_Return(bench1,shift(bench1,1))
b2_ret = Normal_Return(bench2,shift(bench2,1))
fund_ret = Normal_Return(fund,shift(fund,1))

act_ret_1 = fund_ret - b1_ret
act_ret_2 = fund_ret - b2_ret

act_ret_1_ave = np.average(act_ret_1[~np.isnan(act_ret_1)])
act_ret_2_ave = np.average(act_ret_2[~np.isnan(act_ret_2)])

TE_1 = Tracking_error(len(fund), act_ret_1[~np.isnan(act_ret_1)], act_ret_1_ave)
TE_2 = Tracking_error(len(fund), act_ret_2[~np.isnan(act_ret_2)], act_ret_2_ave)

MATE_1 = mean_adjusted_tracking_error(len(fund),act_ret_1[~np.isnan(act_ret_1)]) 
MATE_2 = mean_adjusted_tracking_error(len(fund),act_ret_2[~np.isnan(act_ret_2)])

MATE_2

##}

