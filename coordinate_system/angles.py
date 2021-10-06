"""Module convert angle"""
from typing import Union
import numpy as np

__all__ = ['degrees_to_rad', 'minutes_to_rad', 'seconds_to_rad', 'angle_to_rad']


def degrees_to_rad(degrees: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert angles in degrees to radians
    :param degrees: angle in degrees
    :return: angle in radians
    """
    return degrees / 180 * np.pi


degrees_to_rad_multi = degrees_to_rad  # pylint: disable=invalid-name


def minutes_to_rad(amin: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert minutes to radians
    :param amin: angle in minutes
    :return: angle in radians
    """
    return amin / 60 / 180 * np.pi


def seconds_to_rad(asec: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert seconds to radians
    :param asec: angle in seconds
    :return: angle in radians
    """
    return asec / 3600 / 180 * np.pi


def angle_to_rad(
        degrees: Union[float, np.ndarray] = 0,
        amin: Union[float, np.ndarray] = 0,
        asec: Union[float, np.ndarray] = 0
) -> Union[float, np.ndarray]:
    """
    Convert angles in degrees, minutes and seconds to radians
    :param degrees: angle in degrees
    :param amin: angle in minutes
    :param asec: angle in seconds
    :return: angle in radians
    """
    return (degrees + amin / 60 + asec / 3600) / 180 * np.pi
