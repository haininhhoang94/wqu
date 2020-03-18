import numpy as np

def arithmetic_mean(numpy_array):
    result = sum(numpy_array) / numpy_array.count()
    return result

def geometric_mean(numpy_array):
    result = np.prod(numpy_array)
    return result