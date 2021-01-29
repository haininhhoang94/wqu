# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# %%
X, y = make_blobs(n_samples=200, centers=4, cluster_std=0.5, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)
# %%
k_means = KMeans(n_clusters=4)
k_means.fit(X)
# %%
y_pred = k_means.predict(X)
centers = k_means.cluster_centers_
colors = ["r", "g", "y", "b"]
# %%
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50)
for i, j in zip(centers, colors):
    plt.scatter(i[0], i[1], s=200, c=j, marker="s")
plt.show()
# %%
jgfkldjgklfdjglkdfjgkldfjglkfdjglkdfjlkgjfdlkgjdflkgjdfklgjdflkjgfkdljgklfdjgklfdjgkldfjgklfdjgfdkljgfdl
# %%
# %%
# %%
# %%
