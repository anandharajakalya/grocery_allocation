"""
Module to setup things required to run the project
"""
import os
from os.path import dirname
import sys


PROJECT_ROOT = dirname(dirname(dirname(os.path.abspath(__file__))))

CONFIG_LOCAL = os.path.join('core', 'model', 'placment_id_values')
PARAMETER_LOCAL = os.path.join('core', 'model', 'parameters')

PATH_SAMPLE_DATA_FILES = os.path.join(PROJECT_ROOT, 'data')
PATH_TEMP_DIR = os.path.join(PROJECT_ROOT, 'core', 'tmp')
PATH_CONFIG_DIR = os.path.join(PROJECT_ROOT, CONFIG_LOCAL)
PATH_MODEL_PARAMS_DIR = os.path.join(PROJECT_ROOT, PARAMETER_LOCAL)


def setup_sys_path():
    """
    Add required system path for the project
    """
    print('=== Adding sys path')
    sys.path.append(PROJECT_ROOT)
