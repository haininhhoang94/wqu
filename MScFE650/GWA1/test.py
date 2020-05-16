#%%
import os
os.getcwd()
#%%
import pandas as pd
import numpy as np
import chow_test
#%%
data = pd.read_csv('/home/haininhhoang94/Projects/wqu/MScFE650/GWA1/Chow_test_example/chow_test/financial_time_series.csv')
data.head()
#%%
y1 = data[data['Year'] < 1980]['LogEqPrem']
x1 = data[data['Year'] < 1980]['BookMarket']
y2 = data[data['Year'] >= 1980]['LogEqPrem']
x2 = data[data['Year'] >= 1980]['BookMarket']
#%%
f_test = chow_test.f_value(y1, x1, y2, x2)
print(f_test)
#%%
p_val = chow_test.p_value(y1, x1, y2, x2)
print(p_val)
#%%
#%%
#%%
#%%
