#%%
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import random

#%%
r0 = 0.065
alpha = 0.3
b = 0.09
sigma = 0.03

def fun_1(r,t1,t2):
    return np.exp(-alpha*(t2-t1))*r + b*(1-np.exp(-alpha*(t2-t1)))

def fun_2(t1,t2):
    return (sigma**2)*(1-np.exp(-2*alpha*(t2-t1)))/(2*alpha)

np.random.seed(0)

n_years = 15
n_simulations = 10

t = np.array(range(0,n_years+1))

Z = norm.rvs(size = [n_simulations,n_years])
r_sim = np.zeros([n_simulations,n_years+1])
r_sim[:,0] = r0

for i in range(n_years):
    r_sim[:,i+1] = fun_1(r_sim[:, i],t[i],t[i+1]) + np.sqrt(fun_2(t[i],t[i+1]))*Z[:, i]
    
var_1 = r0*np.exp(-alpha*t)+b*(1-np.exp(-alpha*t))

# %%
# Plotting the result
a = r_sim.shape()