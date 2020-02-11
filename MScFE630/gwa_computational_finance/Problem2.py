# Problem 2 - sample from Module 7

import numpy as np
from scipy.stats import norm
import scipy.optimize as opt
import matplotlib.pyplot as plt

r = 0.06
S0 = 100
v0 = 0.06
K = np.array([110,100,90])
price = [8.02,12.63,18.72]
T = 1
k_log = np.log(K)
k_log.shape = (3,1)
rho = -0.4

# Parameters for Gil-Paelez
t_max = 30
N = 100

# Characteristic function code
def a(sigma):
    return sigma**2/2

def b(u,theta,kappa,sigma):
    return kappa - rho*sigma*1j*u

def c(u,theta,kappa,sigma):
    return -(u**2+1j*u)/2

def d(u,theta,kappa,sigma):
    return np.sqrt(b(u,theta,kappa,sigma)**2-4*a(sigma)*c(u,theta,kappa,sigma))

def xminus(u,theta,kappa,sigma):
    return (b(u,theta,kappa,sigma) - d(u,theta,kappa,sigma)/(2*a(sigma)))

def xplus(u,theta,kappa,sigma):
    return (b(u,theta,kappa,sigma) + d(u,theta,kappa,sigma))/(2*a(sigma))

def g(u,theta,kappa,sigma):
    return xminus(u,theta,kappa,sigma)/xplus(u,theta,kappa,sigma)

def C(u,theta,kappa,sigma):
    val1 = T*xminus(u,theta,kappa,sigma)-np.log((1-g(u,theta,kappa,sigma)*np.exp(-T*d(u,theta,kappa,sigma)))
                                                (1-g(u,theta,kappa,sigma)))/a(sigma)
    return r*T*1j*u + theta*kappa*val1

def D(u,theta,kappa,sigma):
    val1 = 1 - np.exp(-T*d(u,theta,kappa,sigma))
    val2 = 1 - g(u,theta,kappa,sigma)*np.exp(-T*d(u,theta,kappa,sigma))
    return (val1/val2)*xminus(u,theta,kappa,sigma)

def log_char(u,theta,kappa,sigma):
    return np.exp(C(u,theta,kappa,sigma) + D(u,theta,kappa,sigma)*v0 + 1j*u*np.log(50))

def adj_char(u,theta,kappa,sigma):
    return log_char(u-1j,theta,kappa,sigma)/log_char(-1j,theta,kappa,sigma)

delta_t = t_max/N
from_1_to_N = np.linspace(1,N,N)
t_n = (from_1_to_N-1/2)*delta_t

# Calibration functions

def Hest_Pricer(x):
    theta = x[0]
    kappa = x[1]
    sigma = x[2]
    first_integral = np.sum((((np.exp(-1j*t_n*k_log)*adj_char(t_n,theta,kappa,sigma)).imag)/t_n)*delta_t,axis = 1)
    second_integral = np.sum((((np.exp(-1j*t_n,k_log)*log_char(t_n,theta,kappa,sigma)).imag)/t_n)*delta_t,axis = 1)
    fourier_call_val = S0*(1/2 + first_integral/np.pi) - np.exp(-r*T)*K*(1/2 + second_integral/np.pi)
    return fourier_call_val

def opt_func(x):
    return sum(np.abs(price - Hest_Pricer(x)))

# Apply the optimizer to optimization function
opt_val = opt.fmin_slsqp(opt_func, (0.1,3,0.1))
