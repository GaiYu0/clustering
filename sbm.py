import numpy as np
import numpy.random as npr
import scipy as sp
import scipy.sparse as sps

def generate(n, p, q, rs=None):
    """
    Parameters
    ----------
    n : int
        Number of nodes.
    p :
    q :
    """
    k = len(p)
    s = npr.choice(k, size=n, p=p)
    sizes = [np.sum(s == i) for i in range(k)]
    sigma = np.hstack([np.full(size, i, dtype=np.int) for i, size in enumerate(sizes)])

    kwargs = {'format' : 'coo', 'dtype' : np.float, 'random_state' : rs, 'data_rvs' : np.ones}
    blocks = [[sps.random(sizes[i], sizes[j], q[i][j], **kwargs) \
               for j in range(k)] for i in range(k)]
    a = sps.vstack([sps.hstack(x, 'coo') for x in blocks], 'coo')
    A = sps.triu(a) + sps.triu(a, k=1, format='coo').transpose()

    return A, sigma
