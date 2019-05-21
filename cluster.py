import argparse
import numpy as np
from sklearn.metrics import f1_score

import algorithms
import sbm

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str)
parser.add_argument('--binary', action='store_true')
parser.add_argument('--c-in', type=float)
parser.add_argument('--c-out', type=float)
parser.add_argument('-k', type=int, help='number of communities')
parser.add_argument('-n', type=int, help='number of nodes')
parser.add_argument('--graph', type=str)
args = parser.parse_args()

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

    s = getattr(algorithms, args.algorithm)(A, args.binary)
    print(s.shape, sigma.shape)
    print(f1_score(sigma, s, average='micro'))
    print(f1_score(sigma, s, average='macro'))