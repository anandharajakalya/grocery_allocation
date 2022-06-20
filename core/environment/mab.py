# -*- coding: utf-8 -*-
""" :class:`MAB`, :class:`MarkovianMAB`, :class:`ChangingAtEachRepMAB`, :class:`IncreasingMAB`, :class:`PieceWiseStationaryMAB` and :class:`NonStationaryMAB` classes to wrap the arms of some Multi-Armed Bandit problems.

Such class has to have *at least* these methods:
- ``draw(armId, t)`` to draw *one* sample from that ``armId`` at time ``t``,
- and ``reprarms()`` to pretty print the arms (for titles of a plot),
- and more, see below.
See https://github.com/SMPyBandits/SMPyBandits/issues/71
"""
from __future__ import division, print_function  # Python 2 compatibility
import numpy as np

class MAB(object):
    """Basic Multi-Arm Bandit Problem"""

    def __init__(self, configuration):
        """New MAB"""
        self.arms = []
        self._sparisity = None

        if isinstance(configuration, dict):
            raise NotImplementedError(
                "To implement configuration load as a dictionary")
        else:
            for arm in configuration:
                self.arms.append(arm)

        # Compute the mean and stats
        # : Means of the arms
        self.means = np.array([arm.mean for arm in self.arms])
        self.nbArms = len(self.arms)  #: Number of arms
        # : Sums of rewards of each arm
        self.rewards = np.array([arm.sum for arm in self.arms])
        # : pulls of the arms
        self.pulls = np.array([arm.size for arm in self.arms])
        self.t = np.sum(self.pulls)  # : total number of days
        self.placement_id = np.array([arm._placement_id for arm in self.arms])
        # Square of rewards for calculating variance
        self.rewardsSquared = np.square(self.rewards)
        self.totalconsumption = np.array([arm.consumption for arm in self.arms])

        if self._sparisity is not None:
            print(" - with 'sparsity' =", self._sparsity)  # DEBUG
        self.maxArm = np.max(self.means)  # : Max mean of arms
        self.minArm = np.min(self.means)  # : Min mean of arms
        # Print lower bound and HOI factor
        # print(
        #     "\nThis MAB problem has: \n - a [Lai & Robbins] complexity constant C(mu) = {:.3g} ... \n
        #     - a Optimal Arm Identification factor H_OI(mu) = {:.2%} ...".format(
        #         self.lowerbound(), self.hoifactor()))  # DEBUG
        # print(" - with 'arms' represented as:", self.reprarms(1, latex=True))
        # # DEBUG

    def hoifactor(self):
        """ Compute the HOI factor H_OI(mu), the Optimal Arm Identification (OI) factor,
        for this MAB problem (complexity). Cf. (3.3) in Navikkumar MODI's thesis,
        "Machine Learning and Statistical Decision Making for Green Radio" (2017)."""
        return sum(a.one_hoi(self.maxArm, a.mean)
                   for a in self.arms if a.mean != self.maxArm) / float(self.nbArms)

    def __repr__(self):
        return "{}(nbArms: {}, arms: {}, minArm: {:.3g}, maxArm: {:.3g})".format(
            self.__class__.__name__, self.nbArms, self.arms, self.minArm, self.maxArm)

    # --- Draw samples

    def draw(self, arm_id, t=1):
        """ Return a random sample from the armId-th arm, at time t. Usually t is not used."""
        return self.arms[arm_id].draw(t)

    def draw_nparray(self, arm_id, shape=(1,)):
        """ Return a numpy array of random sample from the armId-th arm, of a certain shape."""
        return self.arms[arm_id].draw_nparray(shape)

    def draw_each(self, t=1):
        """ Return a random sample from each arm, at time t. Usually t is not used."""
        return np.array([self.draw(armId, t) for armId in range(self.nbArms)])

    def draw_each_nparray(self, shape=(1,)):
        """ Return a numpy array of random sample from each arm, of a certain shape."""
        return np.array([self.draw_nparray(armId, shape)
                         for armId in range(self.nbArms)])

    #
    # --- Helper to compute sets Mbest and Mworst

    def m_best(self, m=1):
        """ Set of M best means."""
        sorted_means = np.sort(self.means)
        return sorted_means[-m:]

    def m_worst(self, m=1):
        """ Set of M worst means."""
        sorted_means = np.sort(self.means)
        return sorted_means[:-m]

    def sum_best_means(self, m=1):
        """ Sum of the M best means."""
        return np.sum(self.m_best(m=m))

    #
    # --- Helper to compute vector of min arms, max arms, all arms

    def get_min_arm(self, horizon=None):
        """Return the vector of min mean of the arms.

        - It is a vector of length horizon.
        """
        return np.full(horizon, self.minArm)
        # return self.minArm  # XXX Nope, it's not a constant!

    def get_max_arm(self, horizon=None):
        """Return the vector of max mean of the arms.

        - It is a vector of length horizon.
        """
        return np.full(horizon, self.maxArm)
        # return self.maxArm  # XXX Nope, it's not a constant!

    def get_max_arms(self, m=1, horizon=None):
        """Return the vector of sum of the M-best means of the arms.

        - It is a vector of length horizon.
        """
        # TODO: Explected type @akalya
        return np.full(horizon, self.sum_best_means(m))

    def get_all_means(self, horizon=None):
        """Return the vector of means of the arms.

        - It is a numpy array of shape (nbArms, horizon).
        """
        # allMeans = np.tile(self.means, (horizon, 1)).T
        all_means = np.zeros((self.nbArms, horizon))
        for t in range(horizon):
            all_means[:, t] = self.means
        return all_means

    #
    # --- Estimate sparsity

    @property
    def sparsity(self):
        """ Estimate the sparsity of the problem, i.e., the number of arms with positive means."""
        # TODO: Unresolved attribute @akalya
        if self._sparsity is not None:
            return self._sparsity
        else:
            return np.count_nonzero(self.means > 0)

    def str_sparsity(self):
        """ Empty string if ``sparsity = nbArms``, or a small string ', $s={}$' if the sparsity is strictly less than
        the number of arm. """
        s, k = self.sparsity, self.nbArms
        assert 0 <= s <= k, "Error: sparsity s = {} has to be 0 <= s <= K = {}...".format(
            s, k)
        # WARNING
        # disable this feature when not working on sparse simulations
        # return ""
        # or bring back this feature when working on sparse simulations
        return "" if s == k else ", $s={}$".format(s)

    #
    # --- Compute lower bounds

    def lowerbound(self):
        r""" Compute the constant :math:`C(\mu)`, for the [Lai & Robbins] lower-bound for this MAB problem (
        complexity), using functions from ``kullback.py`` or ``kullback.so`` (see :mod:`Arms.kullback`). """
        return sum(a.one_lr(self.maxArm, a.mean)
                   for a in self.arms if a.mean != self.maxArm)

    def lowerbound_sparse(self, sparsity=None):
        """ Compute the constant :math:`C(mu)`, for [Kwon et al, 2017] lower-bound for sparse bandits for this MAB
        problem (complexity)

        - I recomputed suboptimal solution to the optimization problem, and found the same as in [["Sparse Stochastic
        Bandits", by J. Kwon, V. Perchet & C. Vernade, COLT 2017](https://arxiv.org/abs/1706.01383)].
        """
        if hasattr(self, "sparsity") and sparsity is None:
            sparsity = self._sparsity
        if sparsity is None:
            sparsity = self.nbArms

        try:
            try:
                from policies.OSSB import solve_optimization_problem__sparse_bandits
            except ImportError:  # WARNING ModuleNotFoundError is only Python 3.6+
                from SMPyBandits.Policies.OSSB import solve_optimization_problem__sparse_bandits
            ci = solve_optimization_problem__sparse_bandits(
                self.means, sparsity=sparsity, only_strong_or_weak=False)
            # now we use these ci to compute the lower-bound
            gaps = [self.maxArm - a.mean for a in self.arms]
            lowerbound = sum(delta * c for (delta, c) in zip(gaps, ci))
        except (ImportError, ValueError, AssertionError):  # WARNING this is durty!
            lowerbound = np.nan
        return lowerbound
