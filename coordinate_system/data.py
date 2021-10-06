"""Module working with paths"""
import os

__UTILS_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.dirname(__UTILS_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")


def data_path(subpath: str) -> str:
    """
    make full path to open file from `data` folder
    :param subpath: path in `data` folder
    :return: full path of file
    """
    return os.path.join(DATA_DIR, subpath)
