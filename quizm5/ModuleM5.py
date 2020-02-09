# %% [markdown]
# Implementing Dupire's Equation

# %%
# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2

# %%
# Variable declaration
S0 = 100
sigma = 0.3
gamma = 0.75
r = 0.1
T = 3

# %%
# Call price under CEV
z = 2 + 1/(1-gamma)

def C(t,K):
    kappa = 2*r/(sigma**2*(1-gamma)*np.exp(2*r*(1-gamma)*t)-1)
    x = kappa*S0**(2*1-gamma))*np.exp(2*r*(1-gamma)*t)
    y = kappa*K**(2*(1-gamma))
    return S0*(1-ncx2.cdf(y,z,x))-K*np.exp(-r*t)*ncx2.cdf(x,z-2,y)

# %%
# Estimating partial derivatives
delta_t = 0.01
delta_K = 0.01
dC_dT = 