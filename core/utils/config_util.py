"""
Utility for setting the placement ids and values
"""
from core.scripts.setup import CONFIG_LOCAL
from core.utils.transformations_util import get_demand_history
from definitions import DATA_SOURCE_DISK, HISTORICAL_DEMAND_CONSUMPTION
from datetime import datetime, date
from core.utils.io_utils import logger


def create_input_config_file(f_name, data_source=DATA_SOURCE_DISK):
    """
    Returns the config file for each of the input
    :param data_source: local or bq
    :param f_name:
    :return: json
    """

    grocery_list = get_demand_history(f_name, data_source=DATA_SOURCE_DISK)
    filtered_df = grocery_list[['id', 'product_type']].drop_duplicates().reset_index(drop=True)


    placement_id_values = []
    for index, row in filtered_df.iterrows():
        placement_id, policies, prices, current, today_date = {}, {}, {}, {}, {}

    # create placement id  file for each experiment

        # define the policies and parameters for each of the clusters
        policies['archtype'] = "UCBVtuned"
        policies['params'] = {}

        # define the prices to consider, for the algorithm
        current['current'] = eval("{1, 2, 3}")
        today_date['value'] = date.today().strftime("%d-%m-%Y")


        #filter columns to be converted as json
        placement_id['experiment'] = row[['id',
                                 'product_type']].to_dict()
        placement_id['policies'] = policies
        placement_id['placement_ids'] = current
        placement_id['date'] = today_date

        placement_id_values.append(placement_id)

    return placement_id_values


def date_to_process():
    """
    Return the date to predict prices
    :return:
    """
    model_params = get_model_parameters()
    if eval(model_params['CUSTOM_DATE']) is None:
        # get today's date
        date = datetime.today().strftime('%Y-%m-%d')
    else:
        date = model_params['CUSTOM_DATE']

    logger.info(f'-- Processing for date {date}')

    if type(date) != str:
        raise TypeError(f'Date should be of type string')

    return date






