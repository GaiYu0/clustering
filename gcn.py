import argparse

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--activation', type=str)
parser.add_argument('--n-feats', type=int, nargs='+')
Parse = utils.Parse(parser)

class Network(nn.Module):
    def __init__(self, n_feats, activation):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(i, j) for i, j in zip(n_feats[:-1], n_feats[1:])])
        self.activation = getattr(F, activation)

    def forward(self, a, x):
        z = x
        for layer in self.layers[:-1]:
            z = self.activation(th.spmm(a, layer(z)))
        z = th.spmm(a, self.layers[-1](z))
        return z
