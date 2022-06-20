# -*- coding: utf-8 -*-
""" Generic index policy.

- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3],
lower = -3, amplitude = 6. """
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np

from core.policies.base_policy import BasePolicy


class IndexPolicy(BasePolicy):
    """ Class that implements a generic index policy."""

    def __init__(self, nb_arms, lower=0., amplitude=1.):
        """ New generic index policy.

        - nbArms: the number of arms,
        - lower, amplitude: lower value and known amplitude of the rewards.
        """
        super(IndexPolicy, self).__init__(
            nb_arms, lower=lower, amplitude=amplitude)
        self.index = np.zeros(nb_arms)  #: Numerical index for each arms

    # --- Start game, and receive rewards

    def start_game(self):
        """ Initialize the policy for a new game."""
        super(IndexPolicy, self).start_game()
        self.index.fill(0)

    def compute_index(self, arm):
        """ Compute the current index of arm 'arm'."""
        raise NotImplementedError(
            "This method computeIndex(arm) has to be implemented in the child class inheriting from IndexPolicy.")

    def compute_all_index(self):
        """ Compute the current indexes for all arms. Possibly vectorized, by default it can *not* be vectorized
        automatically. """
        for arm in range(self.nbArms):
            self.index[arm] = self.compute_index(arm)


    # --- Basic choice() method

    def choice(self):
        r""" In an index policy, choose an arm with maximal index (uniformly at random):

        .. math:: A(t) \sim U(\arg\max_{1 \leq k \leq K} I_k(t)).

        .. warning:: In almost all cases, there is a unique arm with maximal index, so we loose a lot of time with
        this generic code, but I couldn't find a way to be more efficient without loosing generality.
        """
        # I prefer to let this be another method, so child of IndexPolicy only needs to implement it (if they want,
        # or just computeIndex)
        self.compute_all_index()
        # Uniform choice among the best arms
        try:
            # print(self.index)
            return np.random.choice(np.nonzero(self.index == np.max(self.index))[0]),self.index

        except ValueError:
            print("Warning: unknown error in IndexPolicy.choice(): the indexes were {} but couldn't be used to select "
                  "an arm.".format(self.index))
            return np.random.randint(self.nbArms)

    # --- Others choice...() methods

    def choice_with_rank(self, rank=1):
        """ In an index policy, choose an arm with index is the (1+rank)-th best (uniformly at random).

        - For instance, if rank is 1, the best arm is chosen (the 1-st best).
        - If rank is 4, the 4-th best arm is chosen.


        .. note:: This method is *required* for the :class:`PoliciesMultiPlayers.rhoRand` policy.

        """
        if rank == 1:
            return self.choice()
        else:
            assert rank >= 1, "Error: for IndexPolicy = {}, in choiceWithRank(rank={}) rank has to be >= 1.".format(
                self, rank)
            self.compute_all_index()
            sorted_rewards = np.sort(self.index)
            # Question: What happens here if two arms has the same index, being the max? Then it is fair to chose a
            # random arm with best index, instead of aiming at an arm with index being ranked rank
            chosen_index = sorted_rewards[-rank]
            # Uniform choice among the rank-th best arms
            try:
                return np.random.choice(np.nonzero(self.index == chosen_index)[0])
            except ValueError:
                print("Warning: unknown error in IndexPolicy.choiceWithRank(): the indexes were {} but couldn't be "
                      "used to select an arm.".format(
                        self.index))
                return np.random.randint(self.nbArms)

    def choice_from_subset(self, available_arms='all'):
        """ In an index policy, choose the best arm from sub-set availableArms (uniformly at random)."""
        if isinstance(available_arms, str) and available_arms == 'all':
            return self.choice()
        # If availableArms are all arms? XXX no this could loop, better do it here
        # elif len(availableArms) == self.nbArms:
        #     return self.choice()
        elif len(available_arms) == 0:
            print("WARNING: IndexPolicy.choiceFromSubSet({}): the argument availableArms of type {} should not be "
                  "empty.".format(
                    available_arms, type(available_arms)))  # DEBUG
            # WARNING if no arms are tagged as available, what to do ? choose an arm at random, or call choice() as
            # if available == 'all'
            return self.choice()
        else:
            for arm in available_arms:
                self.index[arm] = self.compute_index(arm)
            # Uniform choice among the best arms
            try:
                return available_arms[np.random.choice(
                    np.nonzero(self.index[available_arms] == np.max(self.index[available_arms]))[0])]
            except ValueError:
                return np.random.choice(available_arms)

    def choice_imp(self, nb=1, start_with_choice_multiple=True):
        """ In an index policy, the IMP strategy is hybrid: choose nb-1 arms with maximal empirical averages,
        then 1 arm with maximal index. Cf. algorithm IMP-TS [Komiyama, Honda, Nakagawa, 2016, arXiv 1506.00779]. """
        if nb == 1:
            return np.array([self.choice()])
        else:
            # For first exploration steps, do pure exploration
            if start_with_choice_multiple:
                if np.min(self.pulls) < 1:
                    return self.choice_multiple(nb=nb)
                else:
                    empirical_means = self.rewards / self.pulls
            else:
                empirical_means = self.rewards / self.pulls
                empirical_means[self.pulls < 1] = float('inf')
            # First choose nb-1 arms, from rewards
            sorted_empirical_means = np.sort(empirical_means)
            exploitations = np.random.choice(np.nonzero(
                empirical_means >= sorted_empirical_means[-nb])[0], size=nb - 1, replace=False)
            # Then choose 1 arm, from index now
            available_arms = np.setdiff1d(np.arange(self.nbArms), exploitations)
            exploration = self.choice_from_subset(available_arms)
            # Affect a random location to is exploratory arm
            return np.insert(exploitations, np.random.randint(np.size(exploitations) + 1), exploration)

    def estimated_order(self):
        """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by
        increasing means. """
        self.compute_all_index()
        return np.argsort(self.index)

    def estimated_best_arms(self, m=1):
        """ Return a (non-necessarily sorted) list of the indexes of the M-best arms. Identify the set M-best."""
        assert 1 <= m <= self.nbArms, "Error: the parameter 'M' has to \
         be between 1 and K = {}, but it was {} ...".format(
            self.nbArms, m)  # DEBUG
        # # WARNING this slows down everything, but maybe the only way to make this correct? if np.all(np.isinf(
        # self.index)): # Initial guess: random estimate of the set Mbest choice = np.random.choice(self.nbArms,
        # size=M, replace=False) print("Warning: estimatedBestArms() for self = {} was called with M = {} but all
        # indexes are +inf, so using a random estimate = {} of Mbest instead of the biased [K-M,...,K-1] ...".format(
        # self, M, choice))  # DEBUG return choice else:
        order = self.estimated_order()
        return order[-m:]
