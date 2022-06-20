"""
Reads and process data from different data sources
"""


import os
from core.scripts.setup import PATH_SAMPLE_DATA_FILES
from core.utils.logger_util import get_logger

logger = get_logger()

def write_to_csv(data, f_name, mode, header):
    """
    Appends or over write data to csv depending on parameters
    :param data:
    :param f_name:
    :param mode:
    :param header:
    :return:
    """
    data.to_csv(os.path.join(PATH_SAMPLE_DATA_FILES, f_name)+".csv", mode=mode, header=header,  index=False)
