import numpy as np
from pandas import read_csv


def load_iris(file):    
    """
    Load iris data.
    
    Parameters
    ----------
    file : char
        Name of the file with the iris data
        
    Returns
    -------
    
    data : (n,d) ndarray
        The n, d-dimensional observations.         
    classes : (k,) list
        The k class labels
    labels : (n,)  list
        The class labels of each observation.          
    """
    data = np.genfromtxt(file, dtype=None, delimiter=",")
    dat = np.zeros((1,4))
    labels = np.array([['0']])

    for x in data:
        t = [y for y in x]
        dat = np.vstack((dat,t[:4]))
        labels = np.vstack((labels,t[4]))

    labels = labels.flatten()[1:]
    dat =  dat[1:,:]
    
    cls = set(labels)
    classes = np.array([x for x in cls])

    return dat, classes, labels
    
def load_iris2(file):
    """
    Read in the iris data.
    Convert the class labels from charcters to numbers. Instead of
    classes, 'Iris-setosa', 'Iris-versicolor' and 'Iris-virginica', the classes 
    are labeled, 0, 1, and 2.
    
    Parameters
    ----------
    file : char
        Name of the input file
        
    Returns
    -------
    data : (n,d) ndarray
        The n, d-dimensional observations.         
    classes : (k,) list
        The k class labels
    labels : (n,)  list
        The class labels of each observation.
    
    
    """
    def iris_to_int(name):
        itypes = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
        return itypes[name.split('-')[1]]

    dat = read_csv(file, converters={4: iris_to_int}, delimiter=',').values
    data = dat[:,:-2]
    labels = np.array(dat[:,-1],dtype = int)
    classes = list(set(labels))
    return data, classes, labels
