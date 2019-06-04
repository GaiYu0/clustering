import numpy.random as npr

def random(A, binary):
    if binary:
        n = A.shape[0]
        s = npr.random(2, n)
    else:
        pass
    return s
