#%%
import os
import sys

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# if BASE_DIR not in sys.path:
#     sys.path.insert(0, BASE_DIR)


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mglearn
# import functions.plot as plot

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept: {}".format(lr.intercept_))
print("Training set score: {:.2f}".format(lr.score(X_train,y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test,y_test)))

# plot.linear_regression(X_train, y_train, X_test, y_test, lr.predict(X_test))

#%%
