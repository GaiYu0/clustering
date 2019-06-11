import numpy as np
import numpy.random as npr
import torch as th

def normalize(xx, eps=1e-5):
    """
    Parameters
    ----------
    xx : list of th.Tensor
    """
    x = xx[0]
    mean = th.mean(x, 0, keepdim=True)
    x = x - mean
    std = th.sqrt(th.mean(x * x, 0, keepdim=True)) + eps
    x = x / std
    xx = [x] + [(x - mean) / std for x in xx[1:]]
    return xx

def partition(x, y, pp):
    """
    Parameters
    ----------
    pp :
    """
    sum_pp = sum(pp)
    pp = [p / sum_pp for p in pp]
    mskk = [(y == i) for i in th.sort(th.unique(y))[0]]
    xx = list(map(x.__getitem__, mskk))
    yy = list(map(y.__getitem__, mskk))
    nnn = [[int(p * len(x)) for p in pp[:-1]] for x in xx]
    nnn = [nn + [len(x) - sum(nn)] for nn, x in zip(nnn, xx)]
    xxx = [th.split(x, nn) for x, nn in zip(xx, nnn)]
    yyy = [th.split(y, nn) for y, nn in zip(yy, nnn)]
    return zip(zip(*xxx), zip(*yyy))

def shuffle(x, y):
    idx = th.randperm(len(y))
    return x[idx], y[idx]
