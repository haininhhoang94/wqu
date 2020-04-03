##{
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

def spot_to_par(spot_array):
    # We will create a loop for coupon_final_price and then match with a
    # using solver

    a = 100 # Assume that the default price of bond is 100
    years = len(spot_array)
    result = np.zeros(shape=(years))

    for i in range(1, years):

        def equation(x):
            b = 0

            for j in range(1,i):
                b = b + x*100 / ((1+spot_array[j-1])**j)

            b = b + a / ((1+spot_array[i-1])**i)
            return (b - a)

        x = fsolve(equation, spot_array[i-1])
        result[i - 1] = x

    return result

spot = np.array([0.06,0.07,0.08,0.09,0.1])
spot_to_par(spot)
##}

##{
def equation(x):
    b = 0

    for j in range(1,3):
        b = b + x*100 / ((1+spot[j-1])**j)

    b = b + 100 / ((1+spot[4])**3)
    return (b - 100)

x = fsolve(equation, 0.05)
x
##}
