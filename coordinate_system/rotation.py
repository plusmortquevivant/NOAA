"""Class Rotation"""
import numpy as np
from typing import Union, Sequence
from .cartesian import get_rotation_z_matrix, get_ecliptic_angle
from .dates import epoch_to_gmst, J2000_ERA_BEGIN_JD, epoch_to_jd, JULIAN_CENTURY, epoch_to_jd_multi
from .types import Epoch, Matrix33
from .nutations import Nutation1980


class Rotation:
    """Class Rotation"""
    def __init__(self, nutation: Nutation1980):
        self.nutation = nutation

    def get_matrix(self, epoch: Union[Epoch, np.ndarray]) -> Union[Matrix33, Sequence[Matrix33]]:
        """
        Class create rotation matrix
        :param epoch: date
        :return: rotation matrix
        """
        if isinstance(epoch, Epoch):
            epoch_jd = epoch_to_jd(epoch)
        else:
            epoch_jd = epoch_to_jd_multi(epoch)
        T = (epoch_jd - J2000_ERA_BEGIN_JD) / JULIAN_CENTURY  # pylint: disable=invalid-name
        eps_mean = get_ecliptic_angle(epoch)

        gmst = epoch_to_gmst(epoch)
        d_psi, d_eps = self.nutation.calc_dpsi_deps(T)
        gast = gmst + d_psi * np.cos(eps_mean + d_eps)
        return get_rotation_z_matrix(gast)
