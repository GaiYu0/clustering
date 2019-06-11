import argparse

import numpy as np
import numpy.random as npr

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int)
parser.add_argument('--mus', type=float, nargs='+')
parser.add_argument('--n-train', type=int, nargs='+')
parser.add_argument('--n-val', type=int, nargs='+')
parser.add_argument('--n-test', type=int, nargs='+')
parser.add_argument('--sigmas', type=float, nargs='+')
Parse = utils.Parse(parser)

def inseparable_gaussian(ns, d, mus, sigmas):
    """
    Parameters
    ----------
    ns : list of ints
    mus : list of floats or arrays with shape (1, d)
    sigmas : list of floats arrays with shape (1, d)
    """
    x = np.vstack([mu + sigma * npr.rand(n, d) \
                   for n, mu, sigma in zip(ns, mus, sigmas)])
    y = np.repeat(np.arange(len(mus)), ns)
    return x, y

def load_dataset(n_train, n_val, n_test, d, mus, sigmas):
    x_train, y_train = inseparable_gaussian(n_train, d, mus, sigmas)
    x_val, y_val = inseparable_gaussian(n_val, d, mus, sigmas)
    x_test, y_test = inseparable_gaussian(n_test, d, mus, sigmas)
    return x_train, y_train, x_val, y_val, x_test, y_test
