# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Submission 3: Simulation Asset Price Evolutions and Reprice Risky up-and-out Call Option
# 
# The goal of Submission 3 is to reprice the risky up-and-out call option from Submission 1, but now implementing a non-constant interest rate and local volatility. With the exception of the interest rates and volatilities, you make make the same assumptions as in Submission 1:
# 
# - Option maturnity is one year
# - The option is struck at-the-money
# - The ip-and-out barrier for the option is $150
# - The current share price is $100
# - The current firm value for the counterpartyis $200
# - The counterparty's deb, due in one year, is $175
# - The correlation between the counterparty and the stock is constant at 0.2
# - The recovery rate with the counterparty is 25%
# 
# The local volatility functions for both the stock and the counterparty have the same form as in part 2, namely ... . For the stock ..., and for the counterparty, ..., where ...
# 
# We can simulation the next step in a share price path using the following formula:
# ()
# 
# where Sti is the share price at time ..., ... is the volatility for the period ..., ... is the risk-free interest rate, and $ ... $. The counterparty firm values can be simulated similarly.
# 
# You observe the following zero-coupon bond prices (per $ 100$ nominal) in the market:
# %% [markdown]
# | Maturity | Price  |
# |----------| ------ |
# | 1 month | $99.38 |
# |----------| ------ |
# |2 months|$98.76|
# |3 months|$98.15|
# |4 months|$97.54|
# |5 months|$96.94|
# |6 months|$96.34|
# |7 months|$95.74|
# |8 months|$95.16|
# |9 months|$94.57|
# |10 months|$93.99|
# |11 months|$93.42|
# |12 months|$92.85|
# %% [markdown]
# You are required to use a LIBOR forward rate model to simulate interest rates. The initial values for the LIBOR forward rates need to be calibrated to the market forward rates which can be deduced through the market zero-coupon bond prices given above. This continuously compounded interest rate for $ ... $ at time $ $, is given by the solution to:
# 
# $$ ... $$
# 
# Where $ ... $ is the LIBOR forward rate which applies from $ $ to $ $ at time $ $. Note that these LIBOR rates are updated as you run through the simulation, and so your continuously compounded rates should be as well.
# 
# For this submission, complete the following tasks:
# 
# 1. Using a sample size of 100000, jointly simulate LIBOR forward rates, stock paths, and counterparty firm values. You should simulate the values monthly, and should have LIBOR forward rates applying over one month, starting one month apart, up to maturity. You may assume that the counterparty firm and stock values are uncorrelated with LIBOR forward rates.
# 
# 2. Calculate the one-year discount factor which applies for each simulation, and use this to find first the value of the option for the jointly simulated stock and firm paths with no default risk, and then the value of the option with counterparty default risk (Hint: you may want to use the reshape and ravel attributes of numpy arrays to ensure your dimensions match correctly.

# %%
# European (up-and-out) Option Pricer
#sdasdsa
# import library
import pandas as pd
import numpy as np
import scipy.optimize
from scipy.stats import uniform, norm
from matplotlib import pyplot as plt
from scipy import log, exp, sqrt, stats
from random import gauss

get_ipython().run_line_magic('matplotlib', 'inline')


# %%
np.random.seed(0)

# Main program

# Option parameters:

S = 100.0   # Initial asset price
sigma = 0.3     # Initial volatility
r = 0.08    # 10 year risk free rate
T = 1.0     # Years until maturity
K = 100.0   # Strike price
B = 150.0   # Barrier price
N = 12      # Number of discrete time points (by months)

# delta T
delta_t = T/N

# counterparty firm parameters
Sf = 200    # Initial asset price
vf = 0.3    # Initial volatility
D = 175.0   # Debt
c = 0.2     # Correlation
rr = 0.25   # Recovery rate


# %%
# Analytical price
def analytical_price(S, K, r, v, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    result = S * stats.norm.cdf(d1) - K * exp(-r*T)*stats.norm.cdf(d2)
    return result

# up & out call price analytical option
#def new_analytical_price():
    
# create European up-and-out option object
print("Analytical price", analytical_price(S, K, r, sigma, T))

#actual zero-coupon bond prices & matirities
bond_prices = np.array([100,99.38,98.76,98.15,97.54,96.94,96.34,95.74,95.16,
                       94.57,93.99,93.42,92.85])/100

maturity = np.array([delta_t * n for n in range(N+1)])

# Analytical bond price Close form solution using Vasicek model
def A(t1,t2,alpha):
    return (1-np.exp(-alpha*(t2-t1)))/alpha

def D(t1,t2,alpha,b,sigma):
    val1 = (t2-t1-A(t1,t2,alpha))*(sigma**2/(2*alpha**2)-b)
    val2 = sigma**2*A(t1,t2,alpha)**2/(4*alpha)
    return val1 - val2

def bond_price_fun(r,t,T,alpha,b,sigma):
    return np.exp(-A(t,T,alpha)*r + D(t,T,alpha,b,sigma))

def F(x):
    r0 = x[0]
    alpha = x[1]
    beta = x[2]
    sigma = x[3]
    return np.sum(np.abs(bond_price_fun(r0,0,maturity,alpha,beta,sigma) - bond_prices))


# %%
# Minimizing F
bnds = ((0,0.2),(0,2),(0,0.5),(0,0.2))
opt_val = scipy.optimize.fmin_slsqp(F,(0.05,0.3,0.05,0.03),bounds=bnds)

print(opt_val)

opt_r = opt_val[0]
opt_alpha = opt_val[0]
opt_beta = opt_val[2]
opt_sigma = opt_val[3]

# Calculating model prices and yield
model_prices = bond_price_fun(opt_r,0,maturity,opt_alpha,opt_beta,opt_sigma)

# print the results
print('\nCalibrated values:')
print('Interest rate = {}'.format(opt_r))
print('Alpha = {}'.format(opt_alpha))
print('Beta = {}'.format(opt_beta))
print('Volatility = {}'.format(opt_sigma))

# Ploting result
plt.xlabel("Maturity")
plt.ylabel("Bond Price")
plt.plot(maturity, bond_prices, label = 'Actual Bond Prices')
plt.plot(maturity, model_prices, 'o', label = 'Analytical Vasicek Bond Prices')
plt.legend()
plt.show()


# %%
# Applying the algorithms
n_simulation = 100000
# Replace the t time step to N, which is from the lecture note
# t = N

# Different from the lecture, we won't use Explcit Monte Carlo simulation here

predcorr_forward = np.ones([n_simulation,N])*(model_prices[:-1]-model_prices[1:])/(delta_t*model_prices[1:])
predcorr_capfac = np.ones([n_simulation, N+1])

delta = np.ones([n_simulation, N])*delta_t


# %%
for i in range(1,N):
    Z = sigma*sqrt(delta[:,i:]*norm.rvs(size = [n_simulation,1]))

    mu_initial = np.cumsum(delta[:,i:]*predcorr_forward[:,i:]*sigma**2/(1+delta[:,i:]*predcorr_forward[:,i:]), axis = 1)
    for_temp = predcorr_forward[:,i:]*exp((mu_initial-sigma**2/2)*delta[:,i:]+Z)
    mu_term = np.cumsum(delta[:,i:]*for_temp*sigma**2/(1+delta[:,i:]*for_temp), axis = 1)
    predcorr_forward[:,i:] = predcorr_forward[:,i:]*exp((mu_initial + mu_term - sigma**2)*delta[:,i:]/2+Z)

# Implying capitalisation factors from the forward rates
predcorr_capfac[:,1:] = np.cumprod(1+delta*predcorr_forward,axis = 1)

# Inverting the capitalisation factors to imply bond prices (discount factors)
predcorr_price = predcorr_capfac**(-1)

# taking averages: Forward Rate, Bond Price, Capitalization Factors

#mean Forward Rate
forward_rate = np.mean(predcorr_forward, axis = 0)

#mean Price
predcorr_price = np.mean(predcorr_price, axis = 0)

# mean Capitalization Factors
capfac = np.mean(predcorr_capfac, axis = 0)

# Plot results
plt.xlabel('Maturity')
plt.ylabel('Bond Price')
plt.plot(maturity, bond_prices, label = 'Acutal Bond Prices')
plt.plot(maturity, model_prices, 'o', label = "Analytical Vasicek Bond Prices")
plt.plot(maturity, predcorr_price, 'x', label = "Predictor-Corrector Bond Prices")
plt.legend()
plt.plot()

# %% [markdown]
# # Problem 2

# %%


