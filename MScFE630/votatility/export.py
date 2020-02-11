import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import ncx2

# share info
S0 = 100    # from submission 1
risk_free_rate = 0.08    # from submission 1
sigma_share = 0.3     # from submission 1

# option info
T = 1
K = S0

# Heston Model relevant
v0 = 0.06
kappa = 9
theta = 0.06
rho = -0.4
sigma_vol = sigma_share  # as asked to use the sigma from submission 1

 First we define a few functions used in Hestion model pricing, 
# function names should indicate corresponding quantities in Hestion Model

a = sigma_vol**2/2

def b(u):
    return kappa - rho*sigma_vol*1j*u

def c(u):
    return -(u**2+1j*u)/2

def d(u):
    return np.sqrt(b(u)**2-4*a*c(u))

def xminus(u):
    return (b(u)-d(u))/(2*a)

def xplus(u):
    return (b(u)+d(u))/(2*a)

def g(u):
    return xminus(u)/xplus(u)

def C(u):
    '''Heston Model price.'''
    val = T*xminus(u)-np.log((1-g(u)*np.exp(-T*d(u)))/(1-g(u)))/a
    return risk_free_rate*T*1j*u + theta*kappa*val

def D(u):
    numer = 1-np.exp(-T*d(u))
    denom = 1-g(u)*np.exp(-T*d(u))
    return (numer/denom)*xminus(u)

def phi_M1(u):
    '''Characteristic function of s_T in Heston Model.'''
    return np.exp(C(u) + D(u)*v0 + 1j*u*np.log(S0))

def phi_M2(u):
    return phi_M1(u-1j)/phi_M1(-1j)

# setting integration parameters
t_max = 30
N = 100
log_k = np.log(K)

# generate sample points
delta_t = t_max/N
points = np.linspace(1,N,N)
t_n = (points-1/2)*delta_t

# estimating integrals by midpoint rule
integral_1 = sum((((np.exp(-1j*t_n*log_k)*phi_M2(t_n)).imag)/t_n)*delta_t)
integral_2 = sum((((np.exp(-1j*t_n*log_k)*phi_M1(t_n)).imag)/t_n)*delta_t)

# finally the Europeann call price
call_price = S0*(1/2+integral_1/np.pi) - np.exp(-risk_free_rate*T)*K*(1/2+integral_2/np.pi)

print("Heston Model Call Price: ", call_price)

# parameters for CEV model 
sigma_CEV = 0.3
gamma = 0.75
delta_t = 1/12

# determine N samples of share price upto time t
def share_price_path(t, N):
    '''Generate N share price path upto time t.'''
    n = int(t/delta_t)
    Z = norm.rvs(size =[N, n])
    price_path = np.array([[np.float64(S0)]*(n+1)]*N)
    for i in range(n):
        vol = sigma_CEV*price_path[:,i]**(gamma-1)
        power = (risk_free_rate-vol**2/2)*delta_t+vol*np.sqrt(delta_t)*Z[:,i]
        price_path[:,i+1]=price_path[:,i]*np.exp(power)
    return price_path

# setting random seed for price path sampling
np.random.seed(10)

# sample price path with various sample size
share_price_T = [None]*50  # expectation of share price at time T
vol_share = [None]*50     # volatility of share price

for i in range(1,51):
    samples = share_price_path(T, i*1000)
    share_price_T[i-1] = np.mean(samples[:,-1])
    vol_share[i-1] = np.std(samples[:, -1])/np.sqrt(i*1000)

# ploting expection of share price at time T
plt.plot(np.array(range(1,51))*1000, share_price_T, label="Share Price at T")
plt.plot(np.array(range(1,51))*1000, [a+3*b for a,b in zip(share_price_T, vol_share)] )
plt.plot(np.array(range(1,51))*1000, [a-3*b for a,b in zip(share_price_T, vol_share)] )
plt.xlabel('Sample Size')
plt.ylabel('Share Price')
plt.legend()
plt.show()

# reset the random seed, so that the sampled price paths can be compared to that in Part 2
np.random.seed(10)

# we reuse some code from Part 2
def call_price_and_stddev(t, N):
    '''Calculate vanilla European call option price and standard deviation under CEV model,
    with maturity t, and N sample paths.'''
    smpls = share_price_path(t, N)
    pay_off = np.maximum(smpls[:, -1]-K, 0)
    return np.mean(pay_off), np.std(pay_off)/np.sqrt(N)

# calculating call price, with maturity T
call_price = [None]*50
call_stddev = [None]*50

for i in range(1, 51):
    call_price[i-1], call_stddev[i-1] = call_price_and_stddev(T, i*1000)

# closed-form call option price under CEV
z = 2+1/(1-gamma)
def closed_form_call_price(t):
    '''Call option price under CEV with maturity t'''
    kappa = 2*risk_free_rate/(sigma_CEV**2*(1-gamma)*(np.exp(2*risk_free_rate*(1-gamma)*t)-1))
    x = kappa*S0**(2*(1-gamma))*np.exp(2*risk_free_rate*(1-gamma)*t)
    y = kappa*K**(2*(1-gamma))
    return S0*(1-ncx2.cdf(y,z,x))-K*np.exp(-risk_free_rate*t)*ncx2.cdf(x,z-2,y)

# plot call option price
plt.plot(np.array(range(1,51))*1000, call_price, '.', label="Monte Carlo")
plt.plot(np.array(range(1,51))*1000, [closed_form_call_price(T)]*50, label="Closed-form")
plt.plot(np.array(range(1,51))*1000, [closed_form_call_price(T)+3*s for s in call_stddev])
plt.plot(np.array(range(1,51))*1000, [closed_form_call_price(T)-3*s for s in call_stddev])
plt.xlabel("Sample Size")
plt.ylabel("Call Price")
plt.legend()
plt.show()

# call price is highly correlated with share price, as expected
np.corrcoef(call_price, share_price_T)