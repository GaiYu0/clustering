import numpy as np
import numpy.random as npr

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--activation', type=str)
parser.add_argument('--n-feats', type=int, nargs='+')
Parse = utils.Parse(parser)

def inseparable_gaussian(ns, mus, sigmas):
    """
    Parameters
    ----------
    ns : list of ints
    mus : list of arrays with shape (1, d)
    sigmas : list of arrays with shape (1, d)
    """
    x = np.vstack([mu + sigma * npr.rand(n, *mu[0].shape) \
                   for n, mu, sigma in zip(ns, mus, sigmas)])
    y = np.repeat(np.arange(len(mus)), ns)
    return x, y

def load_dataset(n_train_samples, n_val_samples, n_test_samples, mus, sigmas):
    x_train, y_train = inseparable_gaussian(n_train_samples, mus, sigmas)
    x_val, y_val = inseparable_gaussian(n_val_samples, mus, sigmas)
    x_test, y_test = inseparable_gaussian(n_test_samples, mus, sigmas)
    return x_train, y_train, x_val, y_val, x_test, y_test
