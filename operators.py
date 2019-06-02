import numpy as np
import scipy.sparse as sps

def adjacency(A):
    return A

def degree(A):
    D = sps.diags(np.ravel(np.asarray(A.sum(axis=1))))
    return D
    
def laplacian(A):
    D = degree(A)
    L = D - A
    return L

def normalized_laplacian(A):
    D = degree(A)

def bethe_hessian(A, r):
    D = degree(A)
    H = (r ** 2 - 1) * sps.eye(A.shape[0]) - r * A + D
    return H
