import numpy as np
import numpy.random as npr
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
from scipy.cluster.vq import kmeans2

import operators

def spectral(A, binary):
    D = sps.diags(np.ravel(np.asarray(A.sum(axis=1))))
    L = D - A
    if binary:
        w, v = linalg.eigsh(L, k=2, which='SA')
        s = v[:, 1] > 0
    else:
        pass
    return s
