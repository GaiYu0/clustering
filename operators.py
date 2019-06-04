import numpy as np
import scipy.sparse as sps

def adjacency(A):
    return A

def degree(A):
    D = sps.diags(np.ravel(np.asarray(A.sum(axis=1))))
    return D

def random_walk(A):
    d = degree(A).data.T
    P = A / d
    return P

def laplacian(A):
    D = degree(A)
    L = D - A
    return L

def normalized_laplacian(A):
    d = degree(A).data
    e = d ** -0.5
    L = sps.eye(len(d)) - e.T * A * e  # TODO
    return L

def bethe_hessian(A, r):
    D = degree(A)
    H = (r ** 2 - 1) * sps.eye(A.shape[0]) - r * A + D
    return H
