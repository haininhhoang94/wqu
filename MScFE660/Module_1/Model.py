# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from functools import reduce
import operator

import os
import requests

import pandas as pd
import numpy as np

import holoviews as hv
import hvplot.pandas


# %%
np.random.seed(42)


# %%
hv.extension("bokeh")

# %%
def plot(mu, sigma, samples):
    return (
        pd.Series(np.random.normal(mu, sigma, 1000))
        .cumsum()
        .hvplot(title="Random Walks", label=f"{samples}")
    )


def prod(mu, sigma, samples):
    return reduce(
        operator.mul, list(map(lambda x: plot(mu, sigma, x), range(1, samples + 1)))
    )


# %%
hv.DynamicMap(prod, kdims=["mu", "sigma", "samples"]).redim.range(
    mu=(0, 5), sigma=(1, 10), samples=(2, 10)
).options(width=900, height=400)

# %%
class Accounts:
    def __init__(
        self,
        account_mu=20,
        account_sigma=5,
        account_numbers=100,
        mu=0,
        sigma=0.05,
        margin_mu=0.1,
        momentum_mu=0.025,
        margin_sigma=0.001,
    ):
        # We initialize paramters for our simulations
        self.account_mu = account_mu
        self.account_sigma = account_sigma
        self.account_numbers = account_numbers

        self.margin_mu = margin_mu
        self.momentum_mu = momentum_mu
        self.margin_sigma = margin_sigma

        self.accounts = np.maximum(
            np.random.normal(
                loc=self.account_mu, scale=self.account_sigma, size=self.account_numbers
            ),
            0,
        )
        self.call = self.accounts * np.random.uniform(0.5, 0.7, self.account_numbers)

        self.mu = mu
        self.sigma = sigma

        self.called_accounts_factor = 0

        self.momentum = 0
        self.history = [0, 0, 0, 0, 0]

        return None

    def price(self):
        # Calculate factors
        self.called_accounts_factor = (
            (self.accounts <= self.call).sum()
        ) / self.account_numbers
        self.momentum = self.history[4] - self.history.pop(0)

        # Update paramteres
        self.mu = (
            self.mu
            - self.margin_mu * self.called_accounts_factor
            + self.momentum_mu * self.momentum
        )
        self.sigma = self.sigma + self.margin_sigma * self.called_accounts_factor

        # Update accounts
        self.history.append(np.random.normal(loc=self.mu, scale=self.sigma))
        self.accounts = self.accounts * (
            1
            + self.history[4]
            * np.random.uniform(low=0.5, high=1.5, size=self.account_numbers)
        )

        # Reset called accounts
        reset = (np.random.rand(self.account_numbers) >= 0.3) * (
            self.accounts <= self.call
        )
        self.accounts[reset] = np.random.normal(self.account_mu, self.account_sigma)
        self.call[reset] = self.accounts[reset] * np.random.uniform(
            0.2, 0.5, np.sum(reset)
        )

        return [
            self.history[4],
            self.accounts.sum(),
            self.called_accounts_factor,
            self.momentum,
        ]


# %% [markdown]
# We will run this simulation over 100 days, and produce five different runs of this simulation to get an idea of the possible outcomes this simulation produces.  We will keep track of the Market Returns, the Average Value of our Margin Accounts, the Number of Accounts called at a given point in time, and the effect they will have on the distribution of returns, as well as the momentum effect we will be used to affect the distribution of returns as well.

# %%
def simulation_plots(days=10000, runs=5):
    run = []

    for _ in range(runs):
        a = Accounts()
        prices = pd.DataFrame(
            [a.price() for day in range(days)],
            columns=[
                "Market Returns",
                "Value of Accounts Trading in the Market",
                "The effect of Margin Calls on Supply",
                "Momentum Effect",
            ],
        )
        run.append(prices)

    plot = (
        reduce(operator.mul, [i.iloc[:, 0].hvplot() for i in run])
        + reduce(operator.mul, [i.iloc[:, 1].hvplot() for i in run])
        + reduce(operator.mul, [i.iloc[:, 2].hvplot() for i in run])
        + reduce(operator.mul, [i.iloc[:, 3].hvplot() for i in run])
    )

    return plot.cols(2)


# %%
get_ipython().run_cell_magic(
    "opts",
    "Curve [width=500 height=300]",
    "hv.DynamicMap(simulation_plots, kdims=['days', 'runs']).redim.range(days=(100,500), runs=(5,15)).options(width=900, height=400)",
)

# %% [markdown]
# From these graphs, you can see how volatile returns appear. Unlike white noise, these feature strong violent trends. Margin Calls take place dramatically for almost all participants in the market like a set of dominoes and momentum tends to carry these effects, as traders panic.
#
# Looking at the graphs below, when we compare this simulation against the standard Normal Distribution and a Normal Distribution with equal mean and variance properties. We see noticeably different characteristics. Our models appear both skew and spread out, with far higher probabilities for events well outside the Standard Normal Distribution; we used as our original function.  While it still appears approximately normal, if we compare it to a normal distribution with similar mean and variance characteristics, we do appear to have far higher levels of kurtosis- indicated by the higher mode of the distribution and the longer left tail.  This longer left tail of this distribution seems to indicate additional skewness to the distribution, which may form an interesting characteristic for investors to consider.

# %%
def simulation_prices(days=100, runs=1000, axis=0):
    run = []

    for _ in range(runs):
        a = Accounts()
        prices = pd.DataFrame([a.price()[0] for day in range(days)], columns=["return"])
        run.append(prices)

    output = pd.concat(run, axis=axis)
    output.columns = [f"Run {i+1}" for i in range(output.shape[1])]

    return output


# %%
simulations = simulation_prices()


# %%
get_ipython().run_cell_magic(
    "opts",
    "Overlay [show_title=True] Distribution [height=500, width=1000]",
    "hv.Distribution(np.random.normal(simulations.mean(),simulations.std(),100000), label='Normal') * hv.Distribution(simulations.iloc[:,0], label='Simulation').options(fill_alpha=0.0)",
)


# %%
# Try creating a Q-Q plot for this data youself,
#  to see the differences in these distributions quartiles


# %%
# Use a Kolmogorov–Smirnov or Shapiro Wilks Test
#  and Analyze your hypothesis

# %% [markdown]
# While a majority of existing research appears to indicate markets returns are not random, using simulation we can begin to observe the effects increased leverage and momentum have in transforming these returns into wider tail distributions, with vastly different market characteristics.  In the 1929 Crash, the introduction of cheap and freely available credit is credited as key contributors to the crash - spurring new approaches to finance and legislation which follower.  Markets are perpetually faced with new policy, theory and strategies which affect markets in a plethora of ways.  While it is difficult to predict the effects, these policies may have on markets, using rudimentary assumptions we can use techniques in simulation to try to uncover and predict their outcomes to aid in financial risk management.
#
# Below, I have included some interesting references for students, on papers which try to simulate and discuss the effects of margin on financial markets.  While some argue that Margin increases risk in the markets, amplifying crashes, others argue it also improves efficiency, allowing trading to better act on market information and structures their portfolios given current characteristics of market microstructure.
# %% [markdown]
# ## References
# _Thurner, S., Farmer, J. D. and Geanakoplos, J. (2012) ‘Leverage causes fat tails and clustered volatility’, Quantitative Finance, 12(5), pp. 695–707. doi: 10.1080/14697688.2012.674301._
#
# _Xiong, W. (2001) ‘Convergence trading with wealth effects: An amplification mechanism in financial markets’, Journal of Financial Economics, 62(2), pp. 247–292. doi: 10.1016/S0304-405X(01)00078-2._
#
#

