""" Defines the objective function of the model"""

import numpy as np
discount_factor = 0.9


def discounted_rewards(credit_consumption):
    """
    Calculates the discounted factor
    :param credit_consumption:
    :return:
    """
    credit_consumed = np.array(credit_consumption)
    discounted_rewards = []
    for index in range(credit_consumed.shape[0]):
        cc_discountfactor = credit_consumed[0:index+1] * discount_factor
        p = np.poly1d(cc_discountfactor)
        discounted_rewards.append(p(1-discount_factor))
    return discounted_rewards
