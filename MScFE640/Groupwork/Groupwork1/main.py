#%%
import numpy as np

def arithmetic_mean(numpy_array):
    result = sum(numpy_array) / np.count_nonzero(numpy_array) 
    return result

def geometric_mean(numpy_array):
    

#%%

test_1 = np.array([0.05,0.028,-0.036,0.005,0.012])
arithmetic_mean(test_1)

# %%
test_2 = np.array([])