# %%
from IPython import get_ipython

# %%
### Fill in some code here to print to console "Financial Engineering"
print("Financial Engineering")

# %% [markdown]
# We will start by importing a number of packages. If your environment is setup correctly from the setup notes, it should execute correctly and without any issues.

# %%
# Import Libraries
import os
import requests

import pandas as pd
import numpy as np

import holoviews as hv
import hvplot.pandas


# %%
# Import Plotting Backend
hv.extension("bokeh")

# %% [markdown]
# The data used for these notes is included in the Data Folder if students would like to run this code themselves and analyse the output.  Code has been included to scrape the data directly from the NYSE website themselves; however, this should not be necessary.  On long-term historical data, it is often challending to find consistent price data; however, volumes are readily recorded.  In order to gain insight into the effects of the crash and history of these markets, we will observe this datapoint overtime in order to gain some peak into evolving market regimes.

# %%
date_ranges = [
    [1970, 1979, "dat"],
    [1960, 1969, "dat"],
    [1950, 1959, "dat"],
    [1940, 1949, "dat"],
    [1930, 1939, "dat"],
    [1920, 1929, "prn"],
    [1900, 1919, "dat"],
    [1888, 1899, "dat"],
][::-1]


# %%
# # Download Data

# def get_decade(start = 1920, end = 1929, extension='prn'):
#     "Specify the sparting year of the decade eg. 1900, 2010, 2009"
#     try:
#         link = requests.get(f'https://www.nyse.com/publicdocs/nyse/data/Daily_Share_Volume_{start}-{end}.{extension}')
#         file = os.path.join("..","Data",f"Daily_Share_Volume_{start}-{end}.{extension}")

#         if link.status_code == 404:
#             raise
#         else:
#             with open(file, 'w') as temp_file:
#                 temp_file.write(str(link.content.decode("utf-8")))

#             print(f"Successfully downloaded {start}-{end}")

#     except:
#         print("There was an issue with the download. \n\
# You may need a different date range or file extension. \n\
# Check out https://www.nyse.com/data/transactions-statistics-data-library")

# download_history = [get_decade(decade[0], decade[1], decade[2]) for decade in date_ranges]

# %% [markdown]
# In order to start exploring this data, we are going to import it into a Pandas Dataframe.  Using this Dataframe we can then import it into HoloWiews in order to track specific data points over time and interact with them as needed.

# %%
# Read and format the data
def load_data(start=1920, end=1929, extension="prn"):
    path = os.path.join("Data", f"Daily_Share_Volume_{start}-{end}.{extension}")

    if extension == "prn":
        data = pd.read_csv(path, sep="   ", parse_dates=["Date"], engine="python").iloc[
            2:, 0:2
        ]
        data.loc[:, "  Stock U.S Gov't"] = pd.to_numeric(
            data.loc[:, "  Stock U.S Gov't"], errors="coerce"
        )
        data.Date = pd.to_datetime(data.Date, format="%Y%m%d", errors="coerce")
        data.columns = ["Date", "Volume"]
        return data
    else:
        data = pd.read_csv(path)
        data.iloc[:, 0] = data.iloc[:, 0].apply(lambda x: str(x).strip(" "))
        data = data.iloc[:, 0].str.split(" ", 1, expand=True)
        data.columns = ["Date", "Volume"]
        data.loc[:, "Volume"] = pd.to_numeric(data.loc[:, "Volume"], errors="coerce")
        data.Date = pd.to_datetime(data.Date, format="%Y%m%d", errors="coerce")
        return data


# %%
data = pd.concat(
    [load_data(decade[0], decade[1], decade[2]) for decade in date_ranges], axis=0
)


# %%
data.head()

# %% [markdown]
# Markets are complex and dynamic systems made up of many agents who not only respond to external information, but to the market itself. These agents learn over time and develop complex behaviour through there interactions. As these markets evolve, characteristics can change requiring new strategies in order to keep up with market trends. Markets are dynamic and can me made up of a number of states. Markets can often respond and behave dramatically different during times of crisis, that they do in either Bull or Bear Markets. While price is a significant concern for investor performance, so too is liquidity. In venture capital, a key question asked is around an investment's exit strategy, and for market investors, the ability to rapidly liquidate investments can be the difference between bankruptcy and success. As market information changes, we can often observe the market forces of supply and demand push and pull, as investors rapidly move to buy and sell-off holdings based on their own investment strategies and fast-changing market information. While liquidity, as a concept, is something difficult to directly quantify, for many investors volume can provide an interesting insight over time into changes to market information, demand and supply and liquidity. When volumes are lower than normal that can often signal little changes in market information when volumes are high, information can be changing dramatically, forcing investors to alter their portfolios and investment strategies.
#
# From the diagram below, we plot Volume for the NYSE from 1888 to 1979 over time. It is clear that volumes have increased dramatically over time, with increasing volatility and kurtosis. While we may speculate around the effect of increased market size, computerized trading and even high-frequency trading, it is interesting to note the dramatic changes markets experience during crisis situations.
#
# We see over a period of time, both before and after Black Tuesday, volumes become increasingly volatile as traders seek to price in the drama of new information. The feature of leverage, new to this market crash, forced many traders to alter their positions in the market in hope of settling margin accounts and hold onto trades.

# %%
# Create plotting object
plot_data = hv.Dataset(data, kdims=["Date"], vdims=["Volume"])

# Create scatter plot

black_tuesday = pd.to_datetime("1929-10-29")

vline = hv.VLine(black_tuesday).options(color="#FF7E47")

m = (
    hv.Scatter(plot_data)
    .options(width=700, height=400)
    .redim("NYSE Share Trading Volume")
    .hist()
    * vline
    * hv.Text(
        black_tuesday + pd.DateOffset(months=10), 4e7, "Black Tuesday", halign="left"
    ).options(color="#FF7E47")
)
m


# %%
# Create plotting object
plot_data_zoom = hv.Dataset(
    data.loc[
        (
            (data.Date >= pd.to_datetime("1920-01-01"))
            & (data.Date <= pd.to_datetime("1940-01-01"))
        ),
        :,
    ],
    kdims=["Date"],
    vdims=["Volume"],
)

# Create scatter plot

black_tuesday = pd.to_datetime("1929-10-29")

vline = hv.VLine(black_tuesday).options(color="#FF7E47")

m = (
    hv.Scatter(plot_data_zoom)
    .options(width=700, height=400)
    .redim("NYSE Share Trading Volume")
    .hist()
    * vline
    * hv.Text(
        black_tuesday + pd.DateOffset(months=10), 4e7, "Black Tuesday", halign="left"
    ).options(color="#FF7E47")
)
m

# %% [markdown]
# Using the slider below, you can adjust the Moving Average Smoothing we can apply to this data and the window of Volatility in order to better comprehend changing market properties.

# %%
data


# %%
get_ipython().run_cell_magic(
    "opts",
    "Scatter [width=400 height=200]",
    "\ndata['Quarter'] = data.Date.dt.quarter\n\ndef second_order(days_window):\n    data_imputed = data\n    data_imputed.Volume = data_imputed.Volume.interpolate()\n    \n    return hv.Scatter(pd.concat([data_imputed.Date, data_imputed.Volume.rolling(days_window).mean()], \n                                names=['Date', 'Volumne Trend'], axis=1)\n                      .dropna()).redim(Volume='Mean Trend') + \\\n    hv.Scatter(pd.concat([data_imputed.Date, data_imputed.Volume.rolling(days_window).cov()], \n                         names=['Date', 'Volumne Variance'], axis=1)\n               .dropna()).redim(Volume='Volume Variance').options(color='#FF7E47')\n    \nhv.DynamicMap(second_order,kdims=['days_window']).redim.range(days_window=(7,1000))",
)


# %%
get_ipython().run_cell_magic(
    "opts",
    "Bars [width=400 height=300]",
    "from statsmodels.tsa.stattools import acf, pacf\n\ndef auto_correlations(start_year, window_years):\n    start_year  = pd.to_datetime(f'{start_year}-01-01')\n    window_years = pd.DateOffset(years=window_years)\n    \n    data_window = data\n    data_window = data_window.loc[((data_window.Date>=start_year)\n                                   &(data_window.Date<=(start_year+window_years))),:]\n    \n    return hv.Bars(acf(data_window.Volume.interpolate().dropna()))\\\n                .redim(y='Autocorrelation', x='Lags') +\\\n            hv.Bars(pacf(data_window.Volume.interpolate().dropna()))\\\n                .redim(y='Patial Autocorrelation', x='Lags').options(color='#FF7E47')\n\nhv.DynamicMap(auto_correlations,kdims=['start_year', 'window_years']\n             ).redim.range(start_year=(data.Date.min().year,data.Date.max().year), window_years=(1,25))",
)

# %% [markdown]
# We can model this data in a rudimentary fashion, looking at the partial auto-correlation and auto-correlation present in this data. These properties can vary dramatically over time and provide insight into the variance, efficiency and responsiveness of the market. Many markets in developing economies can feature low levels of liquidity, even for large stocks. With large public investment companies and retail investors, changes in investment strategy can subsume liquidity in the market, as large volumes of trades look to be executed. In these markets, these trades force the price to increase over many days and may result in increases in one or two-day auto-correlation depending on the characteristics of market liquidity. These characteristics of momentum can also form part of investor strategy, or describe some element of market microstructure, but interesting to note from these plots above is how in recent years auto-correlation of volumes has seen radical changes to historical norms. Generally, in these plots above, we can observe some inkling of these properties in the Partial Auto-correlation Plot, which displays a regular 2-day correlation indicative of characteristics of momentum and liquidity.

# %%
# Try filtering the data and computing
# the skewness and kurtosis over different time periods
# using the .kurtosis() and .skew() functions


# %% [markdown]
# ## References
# _Rappoport, P. and White, E. N. (2016) ‘Was There a Bubble in the 1929 Stock Market ? Published by : Cambridge University Press on behalf of the Economic History Association Stable URL : http://www.jstor.org/stable/2122405 Was There a Bubble in the 1929 Stock Market ?’, 53(3), pp. 549–574._
#
#

# %%

