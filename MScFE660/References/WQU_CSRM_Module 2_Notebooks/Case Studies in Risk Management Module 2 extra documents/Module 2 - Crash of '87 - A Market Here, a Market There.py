# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Validation, Interpretation, Implication!

# Validation, interpretation, and implication are three of the most important steps in analyzing any statistical or economic model. Having introduced the Aggregate Supply-Aggregate Demand Model in the previous notes and the impacts of Cost-Push Inflation, the major question is whether this model serves as an accurate description of the economy and its response to changes in oil prices and Monetarist Policy.  In order to evaluate the AS-AD Model, we are going to simulate its response to changes in oil prices and broad money supply using real economic data and compare it to data on real GDP and price-level in order to graphical test for the model's validity.  Based on our historical discussions earlier in this module, we will be comparing Reserve Bank responses to economic stagflation brought on by increasing oil prices.  In order to identify the effects of Monetarist Policy, we will look at the different approaches taken by Reserve Banks in the United States and the United Kingdom.  Much of the data used in this analysis will be sourced from the World Bank, World Development Indicators, available through the [pandas_datareader API](https://pandas-datareader.readthedocs.io/).  Due to changes in the API and its dependencies, students may need to read through the documentation it requires.  A known issue is this API's compatibility issue with the newer version of the Pandas library. A comment has been added on this in the code and a specific change to one of the Pandas Modules has been added to aid in this compatibility.  You will see this across the notes.  
#   
# In some cases, data has been scaled in order to aid in interpretability, as we are concerned with the effect and not the specific value for broad money or oil prices.  Where possible these changes have been flagged and are crucial for their inclusion in later AS-AD Model.  Our libraries remain the same as the previous set of notes, using NumPy, SciPy, HoloViews, and Pandas, with much of the code repeated from before.  

# +
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import iqr
import pandas as pd

import holoviews as hv
import hvplot.pandas
# -

# There is a compatilibility issue with this library \
#and newer versions of Pandas, this is short fix to the problem, \
#if you have issues at this chunk comment it out and you should be fine.  
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.wb as wb

hv.extension('bokeh')
np.random.seed(42)


# +
def P(*args, **kwargs):
    P = np.linspace(-10, 10, 100).reshape(-1,1)
    P = P[P!=0]
    return P

def AS(P=P(), W=0, P_e=1, Z_2=0):
    return P-Z_2

def AD(P=P(), M=0, G=0, T=0, Z_1=0):
    return -P+Z_1


# -

def findIntersection(fun1,fun2,x0):
    return fsolve(lambda x : fun1(x) - fun2(x),x0)


indicators = wb.get_indicators()
indicators.head()

indicators.loc[indicators.id =='NY.GDP.PETR.RT.ZS',:]

countries = wb.get_countries()
countries.head()

# The World Bank data-portal offers a wide array of data points used to benchmark various interventions, economic growth, prosperity, education, and healthcare.  This data is extensive but often inconsistent across countries or years.  For this reason, it some cases, it is required to find appropriate proxies to capture Macroeconomic variables discussed in academia.  For tracking the real cost of oil over time, we will use the reserve banks measure of 'oil rents (% of GDP)' which is defined as "... the difference between the value of crude oil production at regional prices and total costs of production".  Data was sourced between 1970 to 2017 for use in the analysis in order to get a long run understanding of the events which led up to the crash of 1987 and the implication monetary policy and commodity prices may have on markets today.  
#   
# The graph below shows an accurate measure for the changing economic cost of oil experienced by particular countries over time.  The graph details clear spikes in oil rents in both the US and UK, starting at around 1976, at the time of the first recession.  These reach their peak for either country at around 1980 and 1985 for the US and UK, before restoring to long-run norms. Zooming and panning through the data, you can observe greater detail in these movements as they changed over time and better understand the events at the time of the crash.   

# +
# %%opts Curve [width=800, height=450]
oil = wb.download(indicator='NY.GDP.PETR.RT.ZS', country=['USA','GBR'], start=pd.to_datetime('1970', yearfirst=True), end=pd.to_datetime('2017', yearfirst=True))
oil = oil.reset_index().dropna()

oil_unscaled = oil

oil.loc[oil.country=='United States', 'NY.GDP.PETR.RT.ZS'] = (oil.loc[oil.country=='United States', 'NY.GDP.PETR.RT.ZS'] - 
                                                               oil.loc[oil.country=='United States', 'NY.GDP.PETR.RT.ZS'].mean())/\
                                                                iqr(oil.loc[oil.country=='United States', 'NY.GDP.PETR.RT.ZS'])

oil.loc[oil.country=='United Kingdom', 'NY.GDP.PETR.RT.ZS'] = (oil.loc[oil.country=='United Kingdom', 'NY.GDP.PETR.RT.ZS'] - 
                                                               oil.loc[oil.country=='United Kingdom', 'NY.GDP.PETR.RT.ZS'].mean())/\
                                                                iqr(oil.loc[oil.country=='United Kingdom', 'NY.GDP.PETR.RT.ZS'])

oil_plot = oil.iloc[::-1,:].hvplot.line(x='year', y='NY.GDP.PETR.RT.ZS', by='country', title='Scaled Oil rents (% of GDP)')

oil_plot
# -

# In order to track Reserve Bank policy, we need to observe its effect on money markets and on money supply.  For use in these notes, we will be looking at M4 or broad money, defined as "the sum of currency outside banks; demand deposits other than those of the central government; the time, savings, and foreign currency deposits of resident sectors other than the central government; bank and traveler’s checks; and other securities such as certificates of deposit and commercial paper".  While for specific Reserve Bank interventions M0 and M1 money may be more appropriate, to understanding its implication M4 money will be used to understand its effects through the money multiplier effect.  From the graphs below, we can see around the time of 1985 strongly diverging changes in broad money between these two economic.  While the UK sees stabilizing and increasing broad money, the US experiences a drastic decline in broad money as a percent of GDP, which continued well into the 1990s.  

# +
# %%opts Curve [width=800, height=450]
money = wb.download(indicator='FM.LBL.BMNY.GD.ZS', country=['USA','GBR'], start=pd.to_datetime('1970', yearfirst=True), end=pd.to_datetime('2017', yearfirst=True))
money = money.reset_index().dropna()

money_unscaled = money

money.loc[money.country=='United States', 'FM.LBL.BMNY.GD.ZS'] = (money.loc[money.country=='United States', 'FM.LBL.BMNY.GD.ZS'] - 
                                                                  money.loc[money.country=='United States', 'FM.LBL.BMNY.GD.ZS'].mean())/\
                                                                    iqr(money.loc[money.country=='United States', 'FM.LBL.BMNY.GD.ZS'])

money.loc[money.country=='United Kingdom', 'FM.LBL.BMNY.GD.ZS'] = (money.loc[money.country=='United Kingdom', 'FM.LBL.BMNY.GD.ZS'] - 
                                                                  money.loc[money.country=='United Kingdom', 'FM.LBL.BMNY.GD.ZS'].mean())/\
                                                                    iqr(money.loc[money.country=='United Kingdom', 'FM.LBL.BMNY.GD.ZS'])

money_plot = money.iloc[::-1,:].hvplot.line(x='year', y='FM.LBL.BMNY.GD.ZS', by='country', title='Broad money (% of GDP)')

money_plot
# -

# We will be comparing this variable against real GDP or real output using a constant 2010 USD Price-Level.  Given the relationship between real GDP and our AS-AD model, we will prefer to look at real GDP per capita growth due to its scaling and interpretability. From the graphs below, it is clear that a strong correlation exists between these two countries across time, with clear evidence of recessions in 1987 and 1974.  

# +
# %%opts Curve [width=800, height=450]
gdp = wb.download(indicator='NY.GDP.PCAP.KD', country=['USA','GBR'], start=pd.to_datetime('1970', yearfirst=True), end=pd.to_datetime('2013', yearfirst=True))
gdp = gdp.reset_index()

gdp.loc[:,'NY.GDP.PCAP.KD'] = gdp.loc[:,'NY.GDP.PCAP.KD'].pct_change()

gdp = gdp.loc[pd.to_numeric(gdp.year)<=2012,:].dropna()

gdp_plot = gdp.iloc[::-1,:].hvplot.line(x='year', y='NY.GDP.PCAP.KD', by='country', title='GDP per capita growth (constant 2010 US$)')

gdp_plot


# -

# In the interactive plot below we will use scaled values for broad money and oil rents as values of Z_2 and Z_1 for use in our AS-AD model.  Using the slider, we can more these exogenous shocks through time, observing their effect on Price-level and real GDP output.  Using the real output set at equilibria between this price we scale this equilibrium and compare it against our real GDP, shown by the red dot in the right panel of the graph, to analyze the validity of this model for use across a range of applications.  We compare these models for the UK and the US to arrive at some conclusion around the effects of Monetary Policy on the real economy and capital markets.  These models do not take into account all variable but aim to approximate an estimate of these models' predictions.  

def curves_data_UK(year=1971):
    
    oil_z2 = oil.loc[oil.country=='United Kingdom', 'NY.GDP.PETR.RT.ZS'].iloc[::-1]
    oil_z2 = oil_z2 - oil_z2.iloc[0]
    
    money_z2 = money.loc[money.country=='United Kingdom', 'FM.LBL.BMNY.GD.ZS'].iloc[::-1]
    money_z2 = money_z2 -money_z2.iloc[0]
    
    z_2 = oil_z2.iloc[year-1971] -10
    
    z_1= money_z2.iloc[year-1971]-10
    
    as_eq = pd.DataFrame([P(), AS(P=P(), Z_2=0)], index=['Price-Level','Real Output']).T
    ad_eq = pd.DataFrame([P(), AD(P=P(), Z_1=0)], index=['Price-Level','Real Output']).T
    
    as_shock = pd.DataFrame([P(), AS(P=P(), Z_2=z_2+10)], index=['Price-Level','Real Output']).T
    ad_shock = pd.DataFrame([P(), AD(P=P(), Z_1=z_1+10)], index=['Price-Level','Real Output']).T
    
    result = findIntersection(lambda x: AS(P=x, Z_2=z_2+10), lambda x: AD(P=x, Z_1=-z_1-10), 0.0)
    r = result + 1e-4 if result==0 else result
    
    plot = hv.Curve(as_eq, vdims='Price-Level',kdims='Real Output').options(alpha=0.2, color='#1BB3F5') *\
                              hv.Curve(ad_eq, vdims='Price-Level',kdims='Real Output').options(alpha=0.2, color='orange') *\
                              hv.Curve(as_shock, vdims='Price-Level',kdims='Real Output', label='AS').options(alpha=1, color='#1BB3F5') *\
                              hv.Curve(ad_shock, vdims='Price-Level',kdims='Real Output', label='AD').options(alpha=1, color='orange') *\
                              hv.VLine(-result[0]).options(color='black', alpha=0.2, line_width=1) *\
                              hv.HLine(AS(P=-r[0], Z_2=-z_2-10)).options(color='black', alpha=0.2, line_width=1)
    
    gdp_mean = gdp.loc[gdp.country=='United Kingdom', 'NY.GDP.PCAP.KD'].iloc[0]
    gdp_iqr = iqr(gdp.loc[gdp.country=='United Kingdom', 'NY.GDP.PCAP.KD'])
    
    gdp_plot_UK = gdp.loc[gdp.country=='United Kingdom',:].iloc[::-1,:].hvplot.line(x='year', y='NY.GDP.PCAP.KD', title='GDP per capita growth (constant 2010 US$)') *\
    hv.VLine(year).options(color='black') * pd.DataFrame([[(AD(P=r[0], Z_1=z_1+10)*gdp_iqr*0.35+2.5*gdp_mean), year]], columns=['Real Output', 'year']).hvplot.scatter(y='Real Output', x='year',color='red')
                              
    return plot.options(xticks=[0], yticks=[0], title_format="UK Short-Run AS-AD Model") + gdp_plot_UK


# +
# %%opts Curve [width=400, height=400]

hv.DynamicMap(curves_data_UK, kdims=['year'], label="UK Short-Run AS-AD Model")\
.redim.range(year=(1971,2007))


# -

# Looking at the graphs for UK GDP per capita growth and our AS-AD model, it is clear that while our model fails to account for the peaks in GDP per capita growth in 1973 and 1979, it does appear stationary at those points in time indicative of our comparison between real GDP and real GDP growth camputed in this graph.  Overall the model seems to account well for the overall trend, including growth in 1990 in the mid-2000s.  

def curves_data_US(year=1971):
    
    oil_z2 = oil.loc[oil.country=='United States', 'NY.GDP.PETR.RT.ZS'].iloc[::-1]
    oil_z2 = oil_z2 - oil_z2.iloc[0]
    
    money_z2 = money.loc[money.country=='United States', 'FM.LBL.BMNY.GD.ZS'].iloc[::-1]
    money_z2 = money_z2 -money_z2.iloc[0]
    
    z_2 = oil_z2.iloc[year-1971] -10
    
    z_1= -money_z2.iloc[year-1971]-10
    
    as_eq = pd.DataFrame([P(), AS(P=P(), Z_2=0)], index=['Price-Level','Real Output']).T
    ad_eq = pd.DataFrame([P(), AD(P=P(), Z_1=0)], index=['Price-Level','Real Output']).T
    
    as_shock = pd.DataFrame([P(), AS(P=P(), Z_2=z_2+10)], index=['Price-Level','Real Output']).T
    ad_shock = pd.DataFrame([P(), AD(P=P(), Z_1=z_1+10)], index=['Price-Level','Real Output']).T
    
    result = findIntersection(lambda x: AS(P=x, Z_2=z_2+10), lambda x: AD(P=x, Z_1=-z_1-10), 0.0)
    r = result + 1e-4 if result==0 else result
    
    plot = hv.Curve(as_eq, vdims='Price-Level',kdims='Real Output').options(alpha=0.2, color='#1BB3F5') *\
                              hv.Curve(ad_eq, vdims='Price-Level',kdims='Real Output').options(alpha=0.2, color='orange') *\
                              hv.Curve(as_shock, vdims='Price-Level',kdims='Real Output', label='AS').options(alpha=1, color='#1BB3F5') *\
                              hv.Curve(ad_shock, vdims='Price-Level',kdims='Real Output', label='AD').options(alpha=1, color='orange') *\
                              hv.VLine(-result[0]).options(color='black', alpha=0.2, line_width=1) *\
                              hv.HLine(AD(P=-r[0], Z_1=z_1+10)).options(color='black', alpha=0.2, line_width=1)
    
    gdp_mean = gdp.loc[gdp.country=='United States', 'NY.GDP.PCAP.KD'].iloc[0]
    gdp_iqr = iqr(gdp.loc[gdp.country=='United States', 'NY.GDP.PCAP.KD'])
    
    gdp_plot_US = gdp.loc[gdp.country=='United States',:].iloc[::-1,:].hvplot.line(x='year', y='NY.GDP.PCAP.KD', title='GDP per capita growth (constant 2010 US$)') *\
    hv.VLine(year).options(color='black') * pd.DataFrame([[(AD(P=-r[0], Z_1=-z_1-10))*gdp_iqr*0.3+gdp_mean*4, year]], columns=['Real Output', 'year']).hvplot.scatter(y='Real Output', x='year',color='red')
                              
    return plot.options(xticks=[0], yticks=[0], title_format="US Short-Run AS-AD Model") + gdp_plot_US


# +
# %%opts Curve [width=400, height=400]

hv.DynamicMap(curves_data_US, kdims=['year'], label="US Short-Run AS-AD Model")\
.redim.range(year=(1971,2007))
# -

# Similarly, for the US, the AS-AD model appears fairly accurate in tracking overall trends in the data, despite certicular failures at points in time. Despite it simplicity, the AS-AD model is able to capture the important Macroeconomic dymamics providing predictive insight into the effects of global politics and macroeconomic policy on a countries economy.  It is clear that while countries respond differently to Monetary intervention, such interventions play a crucial role in managing economic crisis which we will explore in greater depth in later modules.  
