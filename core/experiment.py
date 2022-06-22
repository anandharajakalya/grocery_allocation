"""
Invoke multiarmed bandit.
"""

from __future__ import division, print_function

import os, sys
from core.environment.mab import MAB
from core.arms import DiscreteArm
from core.scripts.setup import PATH_SAMPLE_DATA_FILES
from definitions import HISTORICAL_DEMAND_CONSUMPTION
from core.utils.logger_util import get_logger
from core.utils.transformations_util import get_historical_demand_by_price
from core.policies import *
Ellipsi=X=[435]
logger = get_logger()


class Experiment:
    """
    Class to wrap an experiment
    """

    def __init__(self, placement_id_values):
        self.placement_id_values = placement_id_values
        self.arms = []

        # Parse and read the attributes
        self.id = self.placement_id_values['experiment']['id']
        self.product_type = self.placement_id_values['experiment']['product_type']

        self.date = placement_id_values['date']['value']

        self.policy_algorithm = getattr(sys.modules[__name__], self.placement_id_values['policies']['archtype'])
        self.policy_parameters = self.placement_id_values['policies']['params']

        #: List of prices in the current experiment
        self.placement_id = self.placement_id_values['placement_ids']['current']

        # --- Setup the arms for each price and the consumption
        self.__initArms__()
        self.environment = MAB(self.arms)
        self.__initPolicies__()


    def __initPolicies__(self):
        self.policy = self.policy_algorithm(
            self.environment.nbArms, **self.policy_parameters)

        self.policy.t = self.environment.t
        self.policy.rewards = self.environment.rewards
        self.policy.pulls = self.environment.pulls
        self.policy.rewardsSquared = self.environment.rewardsSquared
        index, self.calculation = self.policy.choice()
        self.new_placement_id = self.environment.placement_id[index]

        logger.info(f'-- Model chooses placement id {self.environment.placement_id[index]}')
        self.placement_id = "Model choice"


    def __initArms__(self):
        """
        Create the discrete arms using the historical credit consumption
        Return: None  
        """
        if len(self.placement_id) > 0:

            logger.info(f'Grocery : {self.id} ')
            self.num_placement_id_added = 0
            for placement_id in self.placement_id:
                # Query the historical consumption based on the criteria setup in the experiment
                self.consumption = self._historical_consumption(
                    placement_id,
                    self.id,
                    self.product_type,
                    self.date)

                # Create discrete arm for each placement id
                try:

                    # logger.info(f'-- Adding an Arm with placement id {placement_id} for id {self.id}')
                    self.arms.append(DiscreteArm(placement_id, self.consumption))
                    logger.debug(f' -- Values of appended arm {DiscreteArm(placement_id, self.consumption)}')
                    self.num_placement_id_added += 1

                except Exception as e:
                    logger.error(f'Error {e}')

        else:
            raise ValueError('The price list is empty.Add prices')

    def _historical_consumption(self, placement_id, id, product_type, date):
        """
        Returns the historical credit consumption for the price
        :param price:
        :param id:
        :param product_type:
        :param date:
        :return: list
        """

        if os.path.isfile(os.path.join(PATH_SAMPLE_DATA_FILES, HISTORICAL_DEMAND_CONSUMPTION) + ".csv"):
            # Query from the csv file
            previous_rewards = get_historical_demand_by_price(HISTORICAL_DEMAND_CONSUMPTION, id, product_type, date)


            consumption = previous_rewards[placement_id]['consumed']


        return consumption
