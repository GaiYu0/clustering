import numpy as np
import numpy.random as npr
import scipy.sparse as sps
import scipy.sparse.linalg as linalg

def laplacian(A, binary):
    D = sps.diags(np.ravel(np.asarray(A.sum(axis=1))))
    L = D - A
    if binary:
        w, v = linalg.eigsh(L, k=2, which='SA')
        s = v[:, 1] > 0
    else:
        pass
    return s

def random(A, binary):
    if binary:
        n = A.shape[0]
        s = npr.random(2, n)
    else:
        pass
    return s
