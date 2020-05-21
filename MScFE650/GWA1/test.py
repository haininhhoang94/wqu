
#%%
import sys
sys.path.append('/home/haininhhoang94/Projects/wqu/MScFE650/GWA1/')
#%%
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import datetime
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.api as sm
import seaborn as sns
from scipy import stats

#%%
# Import data
aal = pd.read_csv("~/Projects/wqu/MScFE650/GWA1/RawData/AAL.csv", delimiter=',')
aal['Date'] = pd.to_datetime(aal['Date'], format="%d/%m/%Y")
aal['Date_'] = aal['Date']
#  aal.set_index('Date', inplace=True)

#%%
sns.set_style('darkgrid')
sns.lineplot(x='Date_', y='AdjClose', data=aal)
plt.xticks(rotation=30)
plt.show()
#  aal['AdjClose'].plot(figsize=(16/12))


