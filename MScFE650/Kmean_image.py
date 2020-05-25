# %%
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
import seaborn as sns
with sns.axes_style('dark'):
    img = load_sample_image('china.jpg')
    plt.imshow(img)
# %%
print (img.shape)
# Rescacle the color so that they lie btw 0 and 1, then reshape the array to be
# a typical scikit-learn input
img_r = (img / 255).reshape(-1,3)
print (img_r.shape)
# %%
from sklearn.cluster import KMeans
import numpy as np
k_colors = KMeans(n_clusters=3).fit(img_r)
y_pred = k_colors.predict(img_r)
centers = k_colors.cluster_centers_
labels = k_colors.labels_
new_img = k_colors.cluster_centers_[k_colors.labels_]
new_img = np.reshape(new_img, (img.shape))
# %%
fig = plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,2,1,xticks=[],yticks=[],title='Original Image')
ax.imshow(img)
ax=fig.add_subplot(1,2,2,xticks=[],yticks=[],
                   title='Color Compressed Image using K-Means')
ax.imshow(new_img)
plt.show()
# %%
# %%
# %%
# %%
