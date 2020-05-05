#%%
# Collaborative Review Task 1

#%%
import numpy as np # import the numerical linear algebra model
import matplotlib.pyplot as plt

#%%
x_train = np.linspace(0,1,100)
y_train = 0.2*x_train + 1 + 0.01 * np.random.randn(x_train.shape[0])

x_train_1 = np.array([x_train.tolist()]).T

plt.plot(x_train, y_train, 'r.')
plt.show()

#%%
# In here, we know that we need to do a manual linear regression, thus we will
# use the following equation
# Xw = Y
# Using normal equation like the equation above, with transpose and invert
# Matrix

# Building the X array 
one = np.ones((x_train_1.shape[0], 1))
X = np.concatenate((one, x_train_1), axis = 1)
Y = y_train

# Calculate w = [[m, b]]
# left = (X^T . X)
left = np.dot(X.T, X)
# right = (X^T . Y)
right = np.dot(X.T, Y)
# w = invert(left) . right
w = np.dot(np.linalg.pinv(left), right)

#%%
# Solving the problem with scikit-learn
# Import modules from sklearn
from sklearn.linear_model import LinearRegression

# Fit the data to the model
lr = LinearRegression().fit(x_train.reshape(-1,1), y_train)
plt.plot(x_train, y_train,'.', color='r', label='Train Data')
plt.plot(x_train, lr.predict(x_train.reshape(-1,1)), color='m', 
         label='Prediction')
plt.show()

#%%
# Using Trefethen & Bau equation
# In this problem, A is X and is Y.
# It is very confusing in the lecture note that b both stand for Y and intercept
A = X
b = Y
Q,R = np.linalg.qr(A)
# w_1 = np.dot(np.dot(Q.T, b), np.linalg.inv(R))
w_1 = np.linalg.solve(R, np.dot(Q.T, b))

print(Q.shape, '\n')
print('R = ', R, '\n')
print('Q^TQ = '), np.dot(Q.T,Q)
# print(m, b)
 
#%%
# Solving the problem with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.optimizers import Adam

#%%
# Solve the linear least-squares problem using Keras
# Construct and fit your model here
# Use a batch_size=20, epoch=300
model = Sequential([Dense(1, input_shape = (1,), activation='linear')])
# sgd = optimizers.SGD(lr=0.1)
model.compile(loss='mse', optimizer=Adam(0.01))
model.fit(x_train,Y, epochs = 300, batch_size=20)



