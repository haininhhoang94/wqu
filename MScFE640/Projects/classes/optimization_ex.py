# Solving:
# 4x + 3y = 20
# -5x + 9y = 26

# Closed-form/Analytical solution

##{
import numpy as np

A = np.array([[4,3],[-5,9]])
B = np.array([20,26])

X = np.linalg.inv(A).dot(B)
X
##}

# More difficult example
# 4x + 3y + 2z = 25
# -2x + 2y + 3z = -10
# 3x - 5y + 2z = -4

##{
import numpy as np

A = np.array([[4,3,2], [-2,2,3], [3,-5,2]])
B = np.array([25,-10,-4])

X = np.linalg.inv(A).dot(B)
X
##}

# Using solver method

##{

X2 = np.linalg.solve(A,B)
X2
##}
