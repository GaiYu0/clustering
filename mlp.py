import torch.nn as nn

class Network(nn.Module):
    def __init__(self, n_feats, activation):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(i, j) for i, j in zip(n_feats[:-1], n_feats[1:])])
        self.activation = activation

    def forward(self, a, x):
        z = x
        for layer in self.layers[:-1]:
            z = self.activation(layer(z))
        z = self.layers[-1](z)
        return z
