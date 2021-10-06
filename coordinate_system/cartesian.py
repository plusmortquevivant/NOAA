"""Module cartesian coordinate system"""
from typing import Union, Sequence
import math

import numpy as np
from .angles import angle_to_rad
from .types import Epoch, Vector3, Matrix33

from .dates import J2000_ERA_BEGIN_JD, JULIAN_CENTURY, epoch_to_jd, epoch_to_jd_multi


def get_rotation_x_matrix(phi: Union[float, np.ndarray]) -> Union[Matrix33, Sequence[Matrix33]]:
    """
    :param phi: angle to rotate
    :return: rotation matrix around Ox axis
    """
    if isinstance(phi, float):
        res = np.array([
            [1, 0, 0],
            [0, math.cos(phi), math.sin(phi)],
            [0, -math.sin(phi), math.cos(phi)]
        ])
    else:
        res = np.zeros((len(phi), 3, 3))
        res[:, 0, 0] = 1
        sin_ = np.sin(phi)
        cos_ = np.cos(phi)
        res[:, 1, 1] = cos_
        res[:, 2, 2] = cos_
        res[:, 1, 2] = sin_
        res[:, 2, 1] = -sin_
    return res


def get_rotation_y_matrix(phi: Union[float, np.ndarray]) -> Union[Matrix33, Sequence[Matrix33]]:
    """
    :param phi: angle to rotate
    :return: rotation matrix around Oyaxis
    """
    if isinstance(phi, float):
        res = np.array([
            [math.cos(phi), 0, -math.sin(phi)],
            [0, 1, 0],
            [math.sin(phi), 0, math.cos(phi)]
        ])
    else:
        res = np.zeros((len(phi), 3, 3))
        res[:, 1, 1] = 1
        sin_ = np.sin(phi)
        cos_ = np.cos(phi)
        res[:, 0, 0] = cos_
        res[:, 2, 2] = cos_
        res[:, 0, 2] = -sin_
        res[:, 2, 0] = sin_
    return res


def get_rotation_z_matrix(phi: Union[float, np.ndarray]) -> Union[Matrix33, Sequence[Matrix33]]:
    """
    :param phi: angle to rotate
    :return: rotation matrix around Oz axis
    """
    if isinstance(phi, float):
        res = np.array([
            [math.cos(phi), math.sin(phi), 0],
            [-math.sin(phi), math.cos(phi), 0],
            [0, 0, 1],
        ])
    else:
        res = np.zeros((len(phi), 3, 3))
        res[:, 2, 2] = 1
        sin_ = np.sin(phi)
        cos_ = np.cos(phi)
        res[:, 0, 0] = cos_
        res[:, 1, 1] = cos_
        res[:, 0, 1] = sin_
        res[:, 1, 0] = -sin_
    return res


def get_ecliptic_angle(t: Union[Epoch, np.ndarray]) -> Union[float, np.ndarray]:  # pylint: disable=invalid-name
    """
    Return ecliptic angle in rad at time t
    """
    if isinstance(t, Epoch):
        T = (epoch_to_jd(t) - J2000_ERA_BEGIN_JD) / JULIAN_CENTURY  # pylint: disable=invalid-name
    else:
        T = (epoch_to_jd_multi(t) - J2000_ERA_BEGIN_JD) / JULIAN_CENTURY  # pylint: disable=invalid-name
    return (
            angle_to_rad(23, 26, 21.45) -
            angle_to_rad(asec=46.815) * T -
            angle_to_rad(asec=0.0006) * T ** 2 -
            angle_to_rad(asec=0.00181) * T ** 3
    )


def converse_ecliptic_to_equatorial(t: Epoch, vec: Vector3) -> Vector3:  # pylint: disable=invalid-name
    """
    Converse vector in ecliptic coordinates to equatorial
    :param t: time when conversation happens
    :param vec: vector to converse
    :return: conversed vector
    """
    eps = get_ecliptic_angle(t)
    Rx = get_rotation_x_matrix(-eps)  # pylint: disable=invalid-name
    return Rx.dot(vec)
