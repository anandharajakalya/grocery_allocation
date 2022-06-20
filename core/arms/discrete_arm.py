# -*- coding: utf-8 -*-
""" Discretely distributed arm, of finite support."""

from __future__ import division

import numpy as np
from numpy.random import choice


# Local imports
from core.arms.arm import Arm
from core.utils.objective_function_util import discounted_rewards

class DiscreteArm(Arm):
    """Discreet distributed Arm"""

    def __init__(self, placement_id, values):
        assert len(
            values) > 0, "Error: Discrete Arm values dictionary cannot be empty"
        self._placement_id = placement_id
        self._values = values.copy()
        self._lower = min(self._values)
        self._magnitude = max(self._values) - self._lower
        self.mean = np.mean(self._values)
        self.median = np.median(self._values)
        self.sum = np.sum(discounted_rewards(self._values))  # summation of discounted rewards
        self.consumption = np.sum(self._values)
        self.size = len(self._values)

    def draw(self, t=None):
        """Draw one sample"""
        return choice(self._values)

    def draw_nparray(self, shape=(1,)):
        """Draw a numpy array of random samples, of the certain shape"""
        return np.asarray(choice(self._values, replace=True, size=shape))

    def __str__(self):
        return "DiscreteArm"

    def __repr__(self):
        # return "D({})".format(repr(self._values_to_proba))
        return "D({}{}{})".format("{", ", ".join("{:.3g}: {:.3g}".format(self._price, v) for v in self._values), "}")

    @staticmethod
    def kl(x, y):
        """ The kl(x, y) to use for this arm.

        .. warning:: FIXME this is not correctly defined, except for the special case of
        having **only** 2 values, a ``DiscreteArm``
        is NOT a one-dimensional distribution, and so the kl between two distributions is NOT a function of their mean!
        """
        print("WARNING: DiscreteArm.kl({:.3g}, {:.3g}) is not defined, klBern is used but this is WRONG.".format(
            x, y))  # DEBUG
        return kl_bern(x, y)

    # This decorator @property makes this method an attribute,
    # cf. https://docs.python.org/3/library/functions.html#property
    @property
    def lower_amplitude(self):
        """(lower, amplitude)"""
        return 0., 1.

    # Only export and expose the class defined here
    __all__ = ["DiscreteArm"]
