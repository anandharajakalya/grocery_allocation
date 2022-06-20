# -*- coding: utf-8 -*-
""" Base class for any policy."""

from __future__ import division, print_function  # Python 2 compatibility

import numpy as np

# : If True, every time a reward is received, a warning message is displayed if it lies outsides of ``[lower,
# lower + amplitude]``.
CHECKBOUNDS = False


class BasePolicy(object):
    """
    Base class for policies
    """

    def __init__(self, nb_arms, lower=0., amplitude=1.):
        """New Policy"""

        assert nb_arms > 0, "Error: the nbArms parameter of a {} object cannot be  <=0.".format(
            self)
        self.nbArms = nb_arms  #: Number of Arms
        self.lower = lower  #: lower value for the rewards
        assert amplitude > 0, "Error: the amplitude parameter of a {} object cannot be <=0.".format(
            self)
        self.amplitude = amplitude  #: Delta between the min-max of rewards

        # Internal Memory
        self.t = -1  #: Internal time
        #: Number of pulls of each arm
        self.pulls = np.zeros(nb_arms, dtype=int)
        self.rewards = np.zeros(nb_arms)  #: Cumlative rewards for each arm

    def __str__(self):
        return self.__class__.__name__

    def start_game(self):
        """

        :return:
        """
        self.t = 0
        self.pulls.fill(0)
        self.rewards.fill(0)

    # TODO: Make checkbounds function. for now its being ignored. Probably not needed.
    def get_reward(self, arm, reward, price):
        """

        :param arm:
        :param reward:
        :param price:
        :return:
        """
        self.t += 1
        self.pulls[arm] += 1
        # TODO: Make this reward dependant on the type of arm, as we are mostly using discrete arms.
        # TODO: need to do reward & arm. Will also need to ensure that the arm is an integer/float during definition.
        # reward = (reward - self.lower) / self.amplitude
        self.rewards[arm] += reward * price

    # --- Basic choice()

    def choice(self):
        """Not defined. Has to be taken over by each policy"""
        raise NotImplementedError(
            "This method has to be implements in the child class inheriting the BasePolicy.")

    def choice_multiple(self, nb=1):
        """ Not defined"""
        if nb == 1:
            return np.array([self.choice()])
        else:
            raise NotImplementedError(
                "This method has to be implemented in the child class inheriting thr BasePolicy.")
