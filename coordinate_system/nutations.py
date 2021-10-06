"""Class Nutation"""
from typing import Union, Sequence
import numpy as np
from .utils import matrix_multiple_batch
from .cartesian import get_ecliptic_angle, get_rotation_x_matrix, get_rotation_z_matrix
from .data import data_path
from .angles import seconds_to_rad, angle_to_rad
from .dates import epoch_to_jd, J2000_ERA_BEGIN_JD, JULIAN_CENTURY, epoch_to_jd_multi
from .types import Matrix33, Epoch


def load_nut1980_file(path: str) -> (np.ndarray, np.ndarray):
    """load nut1980_file"""
    int_coefs = np.zeros((106, 5))
    float_coefs = np.zeros((106, 4))
    ind = 0
    with open(path, "r") as f_in:
        for line in f_in:
            if not line or line[0] == "#":
                continue
            for i in range(5):
                int_coefs[ind, i] = (int(line[i * 3: i * 3 + 3]))
            for i in range(4):
                float_coefs[ind, i] = (float(line[15 + 8 + i * 10: 15 + 8 + (i + 1) * 10]))
            ind += 1
    if ind != 106:
        raise Exception("Invalid file with coeffs")
    return int_coefs, float_coefs * 1e-4


class Nutation1980:
    """Class Nutation"""
    def __init__(self, filename=None):
        if filename is None:
            filename = "nut_IAU1980.dat"

        path = data_path(filename)
        self.a_p_coefs, self.psi_eps_coefs = load_nut1980_file(path)

    def calc_dpsi_deps(self, times: Union[float, np.ndarray]):  # pylint: disable=invalid-name
        """calc"""
        l, l1, F, D, Omega = self.calc_params(times)  # pylint: disable=invalid-name
        dPsi = 0  # pylint: disable=invalid-name
        dEps = 0  # pylint: disable=invalid-name
        for a_p_coeffs, psi_eps_coeffs in zip(self.a_p_coefs, self.psi_eps_coefs):
            a_p = (
                a_p_coeffs[0] * l +
                a_p_coeffs[1] * l1 +
                a_p_coeffs[2] * F +
                a_p_coeffs[3] * D +
                a_p_coeffs[4] * Omega
            )
            dPsi += (psi_eps_coeffs[0] + psi_eps_coeffs[1] * times) * np.sin(a_p)  # pylint: disable=invalid-name
            dEps += (psi_eps_coeffs[2] + psi_eps_coeffs[3] * times) * np.cos(a_p)  # pylint: disable=invalid-name
        res = (
            seconds_to_rad(dPsi),
            seconds_to_rad(dEps)
        )
        return res

    def calc_params(self, times: Union[float, np.ndarray]):  # pylint: disable=R0201, C0103
        """cal param for time"""
        l = angle_to_rad(134.96340251) + seconds_to_rad(  # pylint: disable=invalid-name # noqa: E741
            1717915923.2178 * times +
            31.8792 * times ** 2 +
            0.051635 * times ** 3 -
            0.00024470 * times ** 4
        )
        l1 = angle_to_rad(357.52910918) + seconds_to_rad(  # pylint: disable=invalid-name
            129596581.0481 * times -
            0.5532 * times ** 2 -
            0.000136 * times ** 3 -
            0.00001149 * times ** 4
        )
        F = angle_to_rad(93.27209062) + seconds_to_rad(  # pylint: disable=invalid-name
            1739527262.8478 * times -
            12.7512 * times ** 2 +
            0.001037 * times ** 3 +
            0.00000417 * times ** 4
        )
        D = angle_to_rad(297.85019547) + seconds_to_rad(  # pylint: disable=invalid-name
            1602961601.2090 * times -
            6.3706 * times ** 2 +
            0.006593 * times ** 3 -
            0.00003169 * times ** 4
        )
        Omega = angle_to_rad(125.04455501) + seconds_to_rad(  # pylint: disable=invalid-name
            -6962890.2665 * times +
            7.4722 * times ** 2 +
            0.007702 * times ** 3 -
            0.00005939 * times ** 4
        )
        return l, l1, F, D, Omega  # pylint: disable=invalid-name

    def get_matrix(self, epoch: Union[Epoch, np.ndarray]) -> Union[Matrix33, Sequence[Matrix33]]:
        """return matrix"""
        if isinstance(epoch, Epoch):
            epoch_jd = epoch_to_jd(epoch)
        else:
            epoch_jd = epoch_to_jd_multi(epoch)
        times = (epoch_jd - J2000_ERA_BEGIN_JD) / JULIAN_CENTURY  # pylint: disable=invalid-name
        dPsi, dEps = self.calc_dpsi_deps(times)  # pylint: disable=invalid-name
        eps_mean = get_ecliptic_angle(epoch)
        if isinstance(epoch, Epoch):
            res = get_rotation_x_matrix(-eps_mean - dEps).dot(get_rotation_z_matrix(-dPsi)).dot(get_rotation_x_matrix(eps_mean))
        else:
            res = matrix_multiple_batch(
                matrix_multiple_batch(
                    get_rotation_x_matrix(-eps_mean - dEps),
                    get_rotation_z_matrix(-dPsi)
                ),
                get_rotation_x_matrix(eps_mean)
            )
        return res
