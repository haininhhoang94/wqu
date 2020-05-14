#%%

#  %matplotlib inline
#  %load_ext autoreload
#  % autoreload 2

import numpy as np
from matplotlib import pylab as plt
from sklearn.decomposition import PCA

# Import different modules for using with the notebook
from IPython.display import display
from IPython.display import Image
from IPython.display import HTML

#%%

# Create synthetic data
X = np.array([[-1, -1],[-2, -1],[-3, -2],[1, 1],[2, 1],[3, 2]])

# Instantiate and fit PCA model
pca = PCA(n_components=2)
pca.fit(X)

print("Percentage of variance explained by each of the selected components:")
print(pca.explained_variance_ratio_)

#%%

#  Now we will move on to the iris data set
# Load data
from sklearn.datasets import load_iris as ld
data, classes, labels = ld.load_iris2('./data/iris/iris_train.txt')

#%%

# Explain the meaning of explained variance, which provided in the file already
# Fit data
pca = PCA(n_components=3)
pca.fit(data)

# Plot
plt.plot(range(0,3), pca.explained_variance_ratio_)
plt.xlabel('Explained Variance')
plt.ylabel('Princial Compoment')
plt.title('Explained Variance Ratio')
plt.show()

#%%
print(pca.components_)
print(pca.explained_variance_ratio_)

#%%

#Projecting the data
display(Image(filename='./Iris_PCA.png'))

#%%
# Beginning problem 1
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#%%
import load_iris as ld
data, classes, labels = ld.load_iris2('./data/iris/iris_train.txt')
#%%
pca = PCA(n_components=2)
pcs = pca.fit_transform(data)

data_pcs = pd.DataFrame(data = pcs, columns=['PC1', 'PC2'])
#%%
data_pcs
#%%
# Plot the data
colors = ("red", "green", "blue")
for pcs, color, label in zip(pcs,colors,labels):
    x, y = pcs
    plt.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=label)
plt.show()
#%%
#  Beginning Face recognition
# Import all faces as a flattened array
from skimage import io
import numpy as np

#%%
ic = io.ImageCollection('./data/att_faces/*/*.png')
ic = np.array(ic)
ic_flat = ic.reshape((len(ic), -1))

# Shape of array
number, m, n = ic.shape
#%%
from ipywidgets import interact

def view_image(n=0):
    """TODO: Docstring for view_image.

    :n: TODO
    :returns: TODO

    """
    plt.imshow(ic[n], cmap='gray', interpolation='nearest')

w = interact(view_image, n=(0, len(ic)-1))
#%%
pca = PCA(n_components=200)
pca.fit(ic_flat)
#%%
#%%

