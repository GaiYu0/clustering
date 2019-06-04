import argparse
import itertools

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import algorithms
import sbm

parser = argparse.ArgumentParser()
parser.add_argument('--binary', action='store_true')
parser.add_argument('--c-in', type=float)
parser.add_argument('--c-out', type=float)
parser.add_argument('-k', type=int, help='number of communities')
parser.add_argument('-n', type=int, help='number of nodes')
parser.add_argument('--graph', type=str)
args = parser.parse_args()

parser.add_argument('--algorithm', type=str)
for x in ['bethe_hessian', 'non_backtracking', 'random', 'spectral']:
    globals()[x] = __import__(x)
    parser.add_argument('--%s-args' % x, action=globals()[x].Parse)

def overlap(s, t, k):
    return (np.mean(s == t) - 1 / k) / (1 - 1 / k)

if __name__ == "__main__":
    if args.graph == 'partitioning':
        k = 2 if args.binary else args.k
        n = args.n
        c_in = args.c_in
        c_out = args.c_out
        p = np.ones(k) / k
        q = np.ones([k, k]) * c_out / n
        q[range(k), range(k)] *= c_in / c_out
        A, sigma = sbm.generate(args.n, p, q)
    elif args.graph == 'coloring':
        pass
    else:
        pass

    # s = getattr(algorithms, args.algorithm)(A, args.binary)
    s = algorithms.bethe_hessian(A, (c_in + c_out) / 2, k)

    for permutation in itertools.permutations(range(k)):
        t = np.copy(s)
        for i, j in enumerate(permutation):
            t[s == i] = j
        print(overlap(sigma, t, k))

    '''
    print(accuracy_score(sigma, s))
    print(f1_score(sigma, s, average='micro'))
    print(f1_score(sigma, s, average='macro'))
    '''
