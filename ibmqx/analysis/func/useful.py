import numpy as np

def gpc(set):
    #Borland Dennis Constraint
    # L6 + L5 >= L4
    # -> 1 - L1 + 1 - L2 >= 1 - L3
    # -> 1 + L3 >= L2 + L1
    set.sort() 
    set = set[::-1]
    if (set[0]+set[1])<=(1+set[2]):
        # GPC conditions are met
        met = True
    else:
        met = False
    return met


def mean_stdv(data_set):
    N = len(data_set)
    mean = np.sum(data_set)/N
    stdv = np.sqrt(np.sum(np.square(data_set-mean))/N)
    return mean, stdv
