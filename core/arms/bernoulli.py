"""Bernoulli distributed arm"""

import numpy as np
from numpy.random import binomial

from core.arms.arm import Arm
from core.arms.kullback import kl_bern


class Bernoulli(Arm):
    """Bernoulli distributed Arm"""

    def __init__(self, probability):
        assert 0 <= probability <= 1
        self.probability = probability
        self.mean = probability

    def draw(self, t=None):
        """

        :param t:
        :return:
        """
        return binomial(1, self.probability)

    def draw_nparray(self, shape=(1,)):
        """

        :param shape:
        :return:
        """
        return np.asarray(binomial(1, self.probability, shape), dtype=float)

    def lower_amplitude(self):
        """

        :return:
        """
        return 0., 1.

    def __str__(self):
        return "Bernoulli"

    def __repr__(self):
        return "B({:.3g})".format(self.probability)

    @staticmethod
    def kl(x, y):
        """ The kl(x, y) to use for this arm."""
        return kl_bern(x, y)

    @staticmethod
    def one_lr(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Bernoulli arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / kl_bern(mu, mumax)


# Only export and expose the class defined here
__all__ = ["Bernoulli"]
