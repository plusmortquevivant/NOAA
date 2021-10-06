"""Class precessinons"""
from typing import Union, Sequence
import numpy as np
from .utils import matrix_multiple_batch
from .angles import angle_to_rad
from .cartesian import get_rotation_z_matrix, get_rotation_y_matrix
from .dates import epoch_to_jd, J2000_ERA_BEGIN_JD, JULIAN_CENTURY, epoch_to_jd_multi
from .types import Epoch, Matrix33


class Precession:
    """Class Precessions"""
    def get_matrix(self, epoch: Union[Epoch, np.ndarray]) -> Union[Matrix33, Sequence[Matrix33]]:  # pylint: disable=R0201, R0801
        """return matrix for rotation"""
        if isinstance(epoch, Epoch):
            epoch_jd = epoch_to_jd(epoch)
        else:
            epoch_jd = epoch_to_jd_multi(epoch)
        T = (epoch_jd - J2000_ERA_BEGIN_JD) / JULIAN_CENTURY  # pylint: disable=invalid-name
        ksi = (
            angle_to_rad(asec=2306.2181) * T +
            angle_to_rad(asec=0.30188) * T ** 2 +
            angle_to_rad(asec=0.017998) * T ** 3
        )
        theta = (
            angle_to_rad(asec=2004.3109) * T -
            angle_to_rad(asec=0.42665) * T ** 2 -
            angle_to_rad(asec=0.041833) * T ** 3
        )
        zeta = ksi + (
            angle_to_rad(asec=0.79280) * T ** 2 +
            angle_to_rad(asec=0.000205) * T ** 3
        )
        if isinstance(epoch, Epoch):
            res = get_rotation_z_matrix(-zeta).dot(
                get_rotation_y_matrix(theta)
            ).dot(
                get_rotation_z_matrix(-ksi)
            )
        else:
            res = matrix_multiple_batch(
                matrix_multiple_batch(get_rotation_z_matrix(-zeta), get_rotation_y_matrix(theta)),
                get_rotation_z_matrix(-ksi)
            )
        return res


class PrecessionGOST(Precession):
    """Class PrecessionsGOST"""
    def get_matrix(self, epoch: Union[Epoch, np.ndarray]) -> Union[Matrix33, Sequence[Matrix33]]:
        """return matrix"""
        if isinstance(epoch, Epoch):
            epoch_jd = epoch_to_jd(epoch)
        else:
            epoch_jd = epoch_to_jd_multi(epoch)
        T = (epoch_jd - J2000_ERA_BEGIN_JD) / JULIAN_CENTURY  # pylint: disable=invalid-name
        ksi = (
            2.650545 + (
                2306.083227 + (0.2988499 + (0.01801828 - (0.000005791 + 0.0000003173 * T) * T) * T) * T) * T
        ) / 206264.806247097
        theta = (
            (2004.191903 - (0.4294934 + (0.04182264 + (0.000007089 + 0.0000001274 * T) * T) * T) * T) * T
            / 206264.806247097
        )
        zeta = (
            (-2.650545 + (
                2306.077181 + (1.0927348 + (0.0182683 - (0.000028596 + 0.0000002904 * T) * T) * T) * T) * T)
            / 206264.806247097
        )
        if isinstance(epoch, Epoch):
            res = get_rotation_z_matrix(-zeta).dot(
                get_rotation_y_matrix(theta)
            ).dot(
                get_rotation_z_matrix(-ksi)
            )
        else:
            res = matrix_multiple_batch(
                matrix_multiple_batch(get_rotation_z_matrix(-zeta), get_rotation_y_matrix(theta)),
                get_rotation_z_matrix(-ksi)
            )
        return res
