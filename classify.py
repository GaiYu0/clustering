import argparse

import numpy as np
from sklearn.metrics import accuracy_score
import torch as th
import torch.optim as optim
import torch.sparse as ths
import torch.nn.functional as F

import data
import gcn
import mlp
import operators
import sbm

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int)
parser.add_argument('--c-in', type=float)
parser.add_argument('--c-out', type=float)
parser.add_argument('--gpu', type=int)
parser.add_argument('--log-every', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--n-feats', type=int)
parser.add_argument('--n-iterations', type=int)
parser.add_argument('--n-layers', type=int)
parser.add_argument('--n-train', type=int)
parser.add_argument('--n-val', type=int)
parser.add_argument('--network', type=str)
parser.add_argument('--op', type=str)
parser.add_argument('--wd', type=float)
args = parser.parse_args()

device = th.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)

x, y = data.load_binary_covtype()
x = x.to(device)
y = y.to(device)
perm = th.randperm(len(x), device=device)
idx_train = perm[:args.n_train]
idx_val = perm[args.n_train : args.n_train + args.n_val]
idx_test = perm[args.n_train + args.n_val:]

mean = th.mean(x, 0, keepdim=True)
x = x - mean
eps = 1e-5
std = th.sqrt(th.mean(x * x, 0, keepdim=True)) + eps
x = x / std

k = 2
n = len(x)
p = [th.sum(y == 0), th.sum(y == 1)]
c_in = args.c_in
c_out = args.c_out
q = np.ones([k, k]) * c_out / n
q[range(k), range(k)] *= c_in / c_out

A, _ = sbm.generate(n, p, q)
op = sps.coo_matrix(getattr(operators, args.op)(A))
idx = th.from_numpy(np.vstack([op.row, op.col])).long()
dat = th.from_numpy(op.data).float()
a = ths.FloatTensor(idx, dat, [n, n]).to(device)

n_feats = [x.shape[1]] + [args.n_feats] * (args.n_layers - 1) + [2]
network = globals()[args.network].Network(n_feats, F.relu).to(device)
optimizer = optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.wd)

for i in range(args.n_iterations):
    z = network(a, x)
    idx_batch = th.randperm(args.n_train, device=device)[:args.bs]
    idx = idx_train[idx_batch]
    ce = F.cross_entropy(z[idx], y[idx])

    if (i + 1) % args.log_every == 0:
        y_bar = th.argmax(z, 1)
        val_acc = accuracy_score(y[idx_val].tolist(), y_bar[idx_val].tolist())
        test_acc = accuracy_score(y[idx_test].tolist(), y_bar[idx_test].tolist())

        placeholder = '0' * (len(str(args.n_iterations)) - len(str((i + 1))))
        caption = '[iteration %s%d]' % (placeholder, i + 1)
        print('%strain: %.3f | val: %.3f | test: %.3f' % (caption, ce, val_acc, test_acc))

    optimizer.zero_grad()
    ce.backward()
    optimizer.step()
