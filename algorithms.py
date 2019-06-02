import numpy as np
import numpy.random as npr
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
from scipy.cluster.vq import kmeans2

import operators

def laplacian(A, binary):
    D = sps.diags(np.ravel(np.asarray(A.sum(axis=1))))
    L = D - A
    if binary:
        w, v = linalg.eigsh(L, k=2, which='SA')
        s = v[:, 1] > 0
    else:
        pass
    return s

def bethe_hessian(A, c, k):
    r = c ** 0.5
    H_plus = operators.bethe_hessian(A, r)
    H_minus = operators.bethe_hessian(A, r)
    w_plus, v_plus = linalg.eigsh(H_plus, k=10, which='SA')
    w_minus, v_minus = linalg.eigsh(H_minus, k=10, which='SA')
    v = np.hstack([v_plus[:, w_plus < 0], v_minus[:, w_minus < 0]])
    _, s = kmeans2(v, k, minit='points')
    return s

def random(A, binary):
    if binary:
        n = A.shape[0]
        s = npr.random(2, n)
    else:
        pass
    return s
