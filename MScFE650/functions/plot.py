import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style(theme='onedork')

def linear_regression(X_train, y_train, X_test, y_test, y_prediction):
    plt.plot(X_train, y_train,'.', color='r', label='Train Data')
    plt.plot(X_test, y_test,'.', color='r', label='Test Data')
    plt.plot(X_test, y_prediction, color='m', label='Prediction')
    plt.legend()
    plt.show()
    return 0


