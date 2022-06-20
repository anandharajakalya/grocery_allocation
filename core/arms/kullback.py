# -*- coding: utf-8 -*-
""" Kullback-Leibler divergence functions and klUCB utilities."""

from math import log

# TODO: Try to make use of Numba JIT


# : Threshold value: everything in [0,1] is truncated to [eps, 1-eps]
eps = 1e-15

# Simple Kullback - Liebler divergece for Bernoulli distribution

# @jit


def kl_bern(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)

    return x * log(x / y) + (1 - x)*log((1 - x) / (1 - y))
