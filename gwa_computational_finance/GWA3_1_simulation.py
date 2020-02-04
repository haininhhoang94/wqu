# Applying the algorithms
n_simulation = 100000

# From module 6
predcorr_forward = np.ones([n_simulation, N]) \
        * (model_prices[:-1]-model_prices[1:]) \
        / (delta_t*model_prices[1:])
predcorr_capfac = np.ones([n_simulation, N+1])
delta = np.ones([n_simulation, N])*delta_t

# calculate the forward rate for each steps from the bond price
for i in range(1, opt.N):
    # generate random numbers follow normal distribution
    Z = opt.sigma*sqrt(delta[:,i:])*norm.rvs(size = [n,1])

    # predictor-corrector Monte Carlo simulation
    mu_initial = np.cumsum(delta[:,i:]*predcorr_forward[:,i:]*opt.sigma**2/(1+delta[:,i:]*predcorr_forward[:,i:]), axis = 1)
    temp = predcorr_forward[:,i:]*exp((mu_initial-opt.sigma**2/2)*delta[:,i:]+Z)
    mu_term = np.cumsum(delta[:,i:]*temp*opt.sigma**2/(1+delta[:,i:]*temp), axis = 1)
    predcorr_forward[:,i:] = predcorr_forward[:,i:]*exp((mu_initial + mu_term - opt.sigma**2)*delta[:,i:]/2+Z)

# implying capitalization factors from the forward rates
predcorr_capfac[:,1:] = np.cumprod(1+delta*predcorr_forward,axis = 1)

# inverting the capitalization factors to imply bond prices (discount factors)
predcorr_price = predcorr_capfac**(-1)

# taking averages: Forward Rate, Bond Price, Capitalization Factors
# mean Forward Rate
opt.forward_rate = np.mean(predcorr_forward,axis = 0)

# mean Price 
predcorr_price = np.mean(predcorr_price,axis = 0)

# mean Capitalization Factors
opt.capfac = np.mean(predcorr_capfac,axis = 0)

# plot results
plt.subplots(figsize=(16, 8))
plt.xlabel('Maturity')
plt.ylabel('Bond Price')
plt.plot(maturity, zcb_prices, label = 'Actual Bond Prices')
plt.plot(maturity, opt.model_prices, 'o', label = 'Calibration Prices')
plt.plot(maturity, predcorr_price,'x',label = "Predictor-Corrector Bond Prices")
plt.legend()
plt.show()