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

# European up-and-out option class

class Option():
    """parameters"""
    def __init__(self, S = 100, Sf = 200.0, v = 0.3, r = 0.08, vf = 0.3, r = 0.08, vf = 0.3, T = 1.0,
                 K = 100.0, B= 150.0, N = 12, D = 175, c = 0.2, rr = 0.25):

        """constructor"""
        """Stock parameter"""

        self.S = S
        self.v = v
        self.r = r
        self.T = T
        self.K = K
        self.B = B
        self.N = N
        self.delta_t = self.T / self.N

        """paramemters of counterparty firm"""
    def analytical_price(self):
        d1 = (log(S/K) + (r*v**2/2)*T)/(self.v*sqrt(self.T))
        d2 = d1 - v * T
        result = analytical_price = S * stats.norm.cdf(d1) - K * exp(-r*T)*stats.norm.cdf(d2)
        return result

# Main program

# Option parameters:

S = 100.0
v = 0.3
r = 0.08
T = 1.0
K = 100.0
B = 150.0
N = 12

# counterparty firm parameters
Sf = 200
vf = 0.3
D = 175.0
c = 0.2
rr = 0.25

# create European up-and-out option object
opt = Option(S,Sf,v,r,vf,T,K,B,N,D,c,rr)
print(opt)
print("Analytical price", opt.analytical_price)

"""Analytical bond price Close form solution using Vasicek model"""
def A(t1,t2,alpha):
    return (1-np.exp(-alpha*(t2-t1)))/alpha

def D(t1,t2,alpha,b,sigma):
    val1 = (t2-t1-A(t1,t2,alpha))*(sigma**2/(2*alpha**2)-b)
    val2 = sigma**2*A(t1,t2,alpha)**2/(4*alpha)
    return val1 - val2

def bond_price(r,t,T,alpha,b,sigma):
    return np.exp(-A(t,T,alpha)*r + D(t,T,alpha,b,sigma))

#actual zero-coupon bond prices & matirities
zcb_prices = np.read_csv

maturity = np.array([opt.delta_t * n for n in range(opt.N+1)])

# difference between the Vasicek Bond Price and the actual Bond Prices

def F(x):
    r0 = x[0]
    alpha = x[1]
    b = x[2]
    sigma = x[3]
    return np.sum(np.abs(bond_price(r0,0,maturity,alpha,b,sigma) - zcb_prices))

# define the boundary for parameters
bounds = ((0,0.2),(0,5),(0,0.5),(0,2))

# use the minimize function in the Scipy package to calibrate parameters
opt_val = scipy.optimize.fmin_slsqp(F,(0.05,0.3,0.05,0.03),bounds=bounds)
opt.r = opt_val[0]
opt.alpha = opt_val[1]
opt.beta = opt_val[2]
opt.sigma = opt.val[3]

# print result

#estimate model bond prices
opt.model_prices = bond_price(opt.r,0,maturity,opt.alpha,opt.beta,opt.sigma)


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


