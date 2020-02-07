# %%

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Share info
S0 = 100
sigma = 0.3
T = 1
r = 0.06

#Algorithm info
N = 2**10
delta = 0.25
alpha = 1.5

def fft(x):
    N = len(x)
    if N == 1:
        return x
    else:
        ek = fft(x[:-1:2])
        ok = fft(x[1::2])
        m = np.array(range(int(N/2)))
        okm = ok*np.exp(-1j*2*np.pi*m/N)
        return np.concatenate((ek+okm,ek-okm))
    
def log_char(u):
    return np.exp(1j*u*(np.log(S0)+(r-sigma**2/2)*T)-sigma**2*T*u**2/2)

def phi_func(v):
    val1 = np.exp(-r*T)*log_char(v-(alpha+1)*1j)
    val2 = alpha**2+alpha-v**2+1j*(2*alpha+1)*v
    return val1/val2

n = np.array(range(N))
delta_k = 2*np.pi*(N*delta)
b = delta_k*(N-1)/2

log_strike = np.linspace(-b,b,N)

x = np.exp(1j*b*n*delta)*phi_func(n*delta)*delta
x[0] = x[0]*0.5
xhat = fft(x).real

fft_call = np.exp(-alpha*log_strike)*xhat/np.pi

# Call price
d_1 = (np.log(S0/np.exp(log_strike))+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
d_2 = d_1 - sigma*np.sqrt(T)

analytic_callprice = S0*norm.cdf(d_1)-np.exp(log_strike)*np.exp(-r*(T))*norm.cdf(d_2)

# %%
