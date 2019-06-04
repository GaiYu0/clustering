from scipy.cluster.vq import kmeans2
import scipy.sparse.linalg as linalg

import operators

def bethe_hessian(A, c, k):
    r = c ** 0.5
    H_plus = operators.bethe_hessian(A, r)
    H_minus = operators.bethe_hessian(A, r)
    w_plus, v_plus = linalg.eigsh(H_plus, k=10, which='SA')
    w_minus, v_minus = linalg.eigsh(H_minus, k=10, which='SA')
    v = np.hstack([v_plus[:, w_plus < 0], v_minus[:, w_minus < 0]])
    _, s = kmeans2(v, k, minit='points')
    return s
