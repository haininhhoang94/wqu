import numpy as np
from scipy.stats import f

def f_value(y1, x1, y2, x2):
    """ In favor of joshualong. f_value function for Chow Break test package

    :y1: Array like y-values for data preceeding the breakpoint
    :x1: Array like x-values for data preceeding the breakpoint
    :y2: Array like y-values for data occuring after the breakpoint
    :x2: Array like x-values for data occuring after the breakpoint
    :return: F-value: Float value of chow break test

    """
    def find_rss(y, x):
        """This is the subfunction to find the residual sum of squares for a 
        given set of data
        :y: Array like y-values for data subset
        :x: Array like x-values for data subset
        :Returns:
            rss: Returns residual sum of squares of the linear equation 
            represented by that data
            length: The number of n terms that the data represents
        """
        # Preparetion with np.one to stack and transpose for linear regression
        A = np.vstack ([x, np.one(len(x))]).T
        # Least squares solution
        rss = np.linalg.lstsq(A, y, rcond=None)[1]
        length = len(y)
        return (rss, length)

    rss_total, n_total = find_rss(np.append(y1, y2), np.append(x1, x2))
    rss_1, n_1 = find_rss(y1, x1)
    rss_2, n_2 = find_rss(y2, x2)
    
    chow_nom = (rss_total - (rss_1 + rss_2)) / 2
    chow_denom = (rss_1 + rss_2) / (n_1 + n_2 - 4)

    return chow_nom / chow_denom

def p_value(y1, x1, y2, x2, **kwargs):
    F = f_value(y1, x1, y2, x2, **kwargs)
    if not F:
        return 1
    df1 = 2
    df2 = len(x1) + len(x2) - 4

    # The survival function (1-cdf) is more precise than using 1-cdf,
    # this helps when p-values are very close to zero.
    # -f.logsf would be another alternative to directly get -log(pval) instead.
    p_val = f.sf(F[0], df1, df2)
    return p_val
