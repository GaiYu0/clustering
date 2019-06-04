import argparse

import torch as th
import torch.nn as nn

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--n-layers', type=int)
Parse = utils.Parse(parser)

class Network(nn.Module):
    def __init__(self, in_feats, out_feats, n_layers):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.n_layers = n_layers

    def forward(self, a, x):
        z = self.linear(x)
        for i in range(self.n_layers):
            z = th.spmm(a, z)
        return z
