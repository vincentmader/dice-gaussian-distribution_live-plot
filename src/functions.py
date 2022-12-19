import numpy as np

from config import NR_OF_SIDES


def gaussian(x, mu, sigma):
    return np.exp(-1/2 * ((x-mu)/sigma)**2) / (np.sqrt(2*np.pi) * sigma)


def constant(x, h):
    return h * np.ones((len(x)))


def triangular(x, mu, slope):
    return -np.abs(x-mu) / slope + 1/NR_OF_SIDES
