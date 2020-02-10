#%%
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.optimize

years = np.linspace(1,10,19)
yield_curve = (years)**(1/5)/75 + 0.04
var_1 = np.exp(-yield_curve*years)

def A(t1,t2,alpha):
    return (1-np.exp(-alpha*(t2-t1)))/alpha

def D(t1,t2,alpha,b,sigma):
    val1 = (t2-t1-A(t1,t2,alpha))*(sigma**2/(2*alpha**2)-b)
    val2 = sigma**2*A(t1,t2,alpha)**2/(4*alpha)
    return val1 - val2

def bond_price_fun(r,t,T,alpha,b,sigma):
    return np.exp(-A(t,T,alpha)*r+D(t,T,alpha,b,sigma))

r0 = 0.08
r1 = 0.04
r2 = 0.05
r3 = 0.07

def F(x):
    a = x[0]
    b = x[1]
    c = x[2]
    return sum(np.abs(bond_price_fun(r3,0,years,a,b,c)-var_1))