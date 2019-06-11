import argparse

import numpy as np
from sklearn.metrics import accuracy_score
import scipy.sparse as sps
import torch as th
import torch.optim as optim
import torch.sparse as ths
import torch.nn.functional as F

import data
import operators
import sbm
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int)
parser.add_argument('--c-in', type=float)
parser.add_argument('--c-out', type=float)
parser.add_argument('--ds', type=str)
parser.add_argument('--eps', type=float, default=1e-5)
parser.add_argument('--gpu', type=int)
parser.add_argument('--log-every', type=int)
parser.add_argument('--n-iterations', type=int)
parser.add_argument('--op', type=str)

parser.add_argument('--ds', type=str)
for x in ['inseparable_gaussian']:
    parser.add_argument('--%s-args' % x.replace('_', '-'), action=__import__(x).Parse)

parser.add_argument('--network', type=str)
for x in ['gcn', 'mlp', 'sgc']:
    parser.add_argument('--%s-args' % x, action=__import__(x).Parse)

parser.add_argument('--optim', type=str)
for x in ['SGD', 'Adam']:
    parser.add_argument('--%s-args' % x, action=getattr(utils, 'Parse%sArgs' % x))

args = parser.parse_args()

device = th.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)

ds = __import__(args.ds).load_dataset(**vars(getattr(args, args.ds + '_args')))
x_train, y_train, x_val, y_val, x_test, y_test = ds

mean = np.mean(x_train, 0, keepdims=True)
x_train = x_train - mean
std = np.sqrt(np.mean(np.square(x_train), 0, keepdims=True)) + args.eps
x_train = x_train / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std

x = th.from_numpy(np.vstack([x_train, x_val, x_test])).to(device)
y = th.from_numpy(np.vstack([y_train, y_val, y_test])).to(device)
idx_train = th.arange(len(x_train)).to(device).to(device)
idx_val = th.arange(len(x_train), len(x_train) + len(x_val)).to(device)
idx_test = th.arange(len(x_train) + len(x_val), len(x)).to(device)

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

network_args = getattr(args, args.network + '_args')
if hasattr(network_args, 'n_feats'):
    if network_args.n_feats is None:
        network_args.n_feats = [x.shape[1], k]
    else:
        network_args.n_feats = [x.shape[1]] + network_args.n_feats + [k]
else:
    network_args.in_feats = x.shape[1]
    network_args.out_feats = k
network = __import__(args.network).Network(**vars(network_args)).to(device)
optim_args = getattr(args, args.optim + '_args')
optimizer = getattr(optim, args.optim)(network.parameters(), **vars(optim_args))

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
