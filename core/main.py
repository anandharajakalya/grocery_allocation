"""
Starting point for the program
"""
from definitions import DATA_SOURCE_DISK, HISTORICAL_DEMAND_CONSUMPTION
from core.experiment import Experiment
from core.utils.config_util import create_input_config_file
from core.utils.transformations_util import get_demand_history
import pandas as pd


def start():
    """
    Invokes the pricing function.
    """

    ### Create the placement id parameters

    placement_ids_values = create_input_config_file(HISTORICAL_DEMAND_CONSUMPTION, data_source=DATA_SOURCE_DISK)

    for id in placement_ids_values:
        experiment = Experiment(id)

if __name__ == '__main__':
    start()





















































