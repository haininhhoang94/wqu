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

##{
from sympy import Eq, Symbol, solve

y = Symbol('y')
eqn = Eq(y*(8.0 - y**3), 8.0)

solve(eqn)
##}

##{
# solve
#x+y^2 = 4
#e^x + xy = 3

from scipy.optimize import fsolve
import math

def equation(p):
    x, y = p
    return (x+y**2-4, math.exp(x) + x*y - 3)

x, y = fsolve(equation, (1,1))
print("{},{}".format(x,y))
##}
