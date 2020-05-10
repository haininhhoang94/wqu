#%%

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris # this will be our data set



#%%
iris = load_iris()
#%%
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Categorical.from_codes(iris.target, iris.target_names)
#%%
X
y
#%%
# Scaling the data and fit transform to 0,0 and set standard deviation to 1
scaler = StandardScaler()
X = scaler.fit_transform(X)

#%%
pca = PCA(n_components=2)
principal_componenets = pca.fit_transform(X)

new_X = pd.DataFrame(data = principal_componenets, columns=['PC1', 'PC2'])
#%%
new_X.head()
#%%
#%%
#%%
