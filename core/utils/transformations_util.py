"""
Utility for data transformations
"""
from core.scripts.setup import PROJECT_ROOT, PATH_SAMPLE_DATA_FILES
from definitions import  DATA_SOURCE_DISK
import pandas as pd
from core.utils.logger_util import get_logger
import os

logger = get_logger()


def get_demand_history(f_name, data_source=DATA_SOURCE_DISK):
    """
    Returns the historical data
    :param data_source: local or bq
    :param f_name:
    :return: dataframe
    """
    data = None
    if data_source == DATA_SOURCE_DISK:
        # print('-- Reading data from {} csv'.format(f_name))
        data = pd.read_csv(os.path.join(PATH_SAMPLE_DATA_FILES, f_name) + ".csv")
    return data


def get_historical_demand_by_price(f_name, id, product_type, date, data_source=DATA_SOURCE_DISK):
    """
    Returns historical credit consumption for each price level
    :param data_source: local or bq
    :param f_name:
    :param cluster_id:
    :param action:
    :param type_code:
    :param property_type:
    :param date
    :return: dict
    """

    data = get_demand_history(f_name, data_source)

    filtered_df = data[(data['id'] == id) &
                       (data['product_type'] == product_type) &
                       (pd.to_datetime(data['date']) <= date)][['placement_id', 'consumption']]
    placementid_demand_dict = filtered_df.groupby('placement_id')['consumption'].apply(list).reset_index(
        name='consumed').set_index('placement_id').T.to_dict()

    return placementid_demand_dict


def query_data(data, filter):
    """
    Filters out data according to the given conditions
    :param filter:
    :param data:
    :return:
    """

    # Query the data
    for key, values in filter.items():
        data = data[data[key].isin(values)]
    return data


def dict_to_dataframe(dict):
    """
    Converts a dictionary to a dataframe
    :param dict:
    :return: data
    """
    data = pd.DataFrame(dict, index=[0])

    return data
