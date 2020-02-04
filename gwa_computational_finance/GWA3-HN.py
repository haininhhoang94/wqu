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

#%matplotlib inline
np.random.seed(0)

# Main program

# Option parameters:

S = 100.0   # Initial asset price
v = 0.3     # Initial volatility
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

# Analytical price
def analytical_price(S, K, r, v, T):
    d1 = (log(S/K) + (r*v**2/2)*T)/(v*sqrt(T))
    d2 = d1 - v * T
    result = S * stats.norm.cdf(d1) - K * exp(-r*T)*stats.norm.cdf(d2)
    return result

# create European up-and-out option object
print("Analytical price", analytical_price(S, K, r, v, T))

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
    b = x[2]
    sigma = x[3]
    return np.sum(np.abs(bond_price_fun(r0,0,maturity,alpha,b,sigma) - bond_price))

#actual zero-coupon bond prices & matirities
zcb_prices = np.array([100,99.38,98.76,98.15,97.54,96.94,96.34,95.74,95.16,
                       94.57,93.99,93.42,92.85])

maturity = np.array([delta_t * n for n in range(N+1)])


# Minimizing F
bnds = ((0,0.2),(0,5),(0,0.5),(0,2))
opt_val = scipy.optimize.fmin_slsqp(F,(0.05,0.3,0.05,0.03),bounds=bnds)
r = opt_val[0]
opt_alpha = opt_val[1]
opt_beta = opt_val[2]
opt_sigma = opt.val[3]

# Calculating model prices and yield
model_prices = bond_price_fun(r,0,maturity,opt_alpha,opt_beta,opt_sigma)

# Ploting result
plt.xlabel("Maturity")
plt.ylabel("Bond Price")
plt.plot(maturity, bond_prices)
plt.plot(maturity, model_prices, 'x')

# define numbers of simulations
n = 100000

# initialize predictor-corrector Monte Carlo Simulation forward rate
predcorr_forward = np.ones([n,opt.N])*(opt.model_prices[:-1]-opt.model_prices[1:])/(opt.delta_t*opt.model_prices[1:])
predcorr_capfac = np.ones([n, opt.N+1])
delta = np.ones([n, opt.N])*opt.delta_t

#calculate the forward rate for each steps from the bond price
for i in range(1,opt.N):
    # generate random numbers follow normal distribution
    Z = opt.sigma*sqrt(delta[:,i:]*norm.rvs(size = [n,1]))

    #predictor-corrector Monte Carlo Simulation
    mu_initial = np.cumsum(delta[:,i:]*predcorr_forward[:,i:]*opt.sigma**2/(1+delta[:,i:]*predcorr_forward[:,i:]), axis = 1)
    temp = predcorr_forward[:,i:]*exp((mu_initial-opt.sigma**2/2)*delta[:,i:]+Z)
    mu_term = np.cumsum(delta[i:]*temp*opt.sigma**2/(1+delta[:,i:]+Z))
    predcorr_forward[:,i:] = predcorr_forward[:,i:]*exp((mu_initial + mu_term - opt.sigma**2)*delta[:,i:]/2+Z)

# implying capitalization factors from the forward rates
predcorr_capfac[:,1:] = np.cumprod(1+delta*predcorr_forward,axis = 1)

# inverting the capitalization factors to imply bond prices (discount factors)
predcorr_price = predcorr_capfac**(-1)

# taking averages: Forward Rate, Bond Price, Capitalization Factors

#mean Forward Rate
opt.forward_rate = np.mean(predcorr_forward, axis = 0)

#mean Price
predcorr_price = np.mean(predcorr_price, axis = 0)

# mean Capitalization Factors
opt.capfac = np.mean(predcorr_capfac, axis = 0)

# plot.....

def simulate(Option, n = 10000, gamma = 0.75):
    Option.stock_paths = []
    Option.firm_paths = []
    Option.payoffs = []
    Option.losses = []

    # Constans calculate outside the loop for optimization

    corr_matrix = np.linalg.cholesky(np.array([[1,Option.c],[Option.c, 1]]))

    for i in range(n):
        stock_path = []
        firm_path = []
        S_j = Option.S_j
        Sf_j = Option.Sf
        for j in range(Option.N):
            stock_path.append(S_j)
            firm_path.append(Sf_j)
            xi = np.matmul(corr_matrix, norm.rvs(size = 2))

            # local volailities
            v_dt = Option.v*(S_j)**(gamma-1)
            vf_dt = Option.vf*(Sf_j)**(gamma-1)

            # continuously compounded interest rate
            Option.r = log(1 + opt.forward_rate[j]*Option.delta_t)/Option.delta_t

            #stock price
            S_j *= exp((Option.r-1/2*(vf_dt**2))*Option.delta_t +
                       v_dt*sqrt(Option.delta_t)*xi[0])
            
            #firm price
            Sf_j *= exp((Option.r-1/2*(vf_dt**2)*Option.delta_t) + 
                        vf_dt*sqrt(Option.delta_t)*xi[1])
            
        # Saving the stock and firm path
        Option.stock_paths.append(stock_path)
        Option.firm_paths.append(firm_path)
        
        # Call payoff
        if np.max(stock_path) < Option.B:
            Option.payoffs.append(np.maximum(stock_path[-1]
                                             Option.K, 0))
        else:
            Option.payoffs.append(0)
            
        # Firm's losses
        if np.min(firm_path) < Option.D:
            Option.losses.apeend(exp(-Option.vf*Option.T)*(1-Option.rr)*Option.payoffs[-1])
        else:
            Option.losses.append(0)
            
        # use payoff vector and discount factor to compute the price of the option
        Vi = exp(-Option.r * Option.T) * np.array(Option.payoffs, dtype=float)
        Option.call = np.mean(Vi)
        Option.call_sd = np.std(Vi) / np.sqrt(n*Option.N)
        
        # estimating CVA
        Option.cva = np.mean(Option.losses)
        Option.cva_sd = np.std(Option.losses) / np.sqrt(n*Option.N)
        
# running simulation
simulate(opt,n)
print("Done")

