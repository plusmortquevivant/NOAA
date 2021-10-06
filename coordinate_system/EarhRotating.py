"""Provides a change of coordinates"""
from typing import Union, Sequence
import numpy as np
from .utils import matrix_multiple_batch
from .types import Epoch, Matrix33
from .precessions import Precession, PrecessionGOST
from .nutations import Nutation1980
from .rotation import Rotation


class EarthRotatingCS:
    """provides a change of coordinates"""

    def __init__(self, rot: Rotation, precession: Precession, nutation: Nutation1980):
        self.rotation = rot
        self.precession = precession
        self.nutation = nutation

    def get_matrix(self, epoch: Union[Epoch, np.ndarray]) -> Matrix33:
        """
        Create rotation matrix
        :param epoch: date
        :return: matrix
        """
        if isinstance(epoch, Sequence):
            epoch = np.array(epoch)
        rot = self.rotation.get_matrix(epoch)
        prec = self.precession.get_matrix(epoch)
        nut = self.nutation.get_matrix(epoch)
        if isinstance(epoch, Epoch):
            res = rot.dot(nut).dot(prec)
        else:
            res = matrix_multiple_batch(matrix_multiple_batch(rot, nut), prec)
        return res

    def get_matrix_fast(self, epoch: Union[Epoch, np.ndarray]) -> Matrix33:
        """

        Возвращает матрицу поворота, рассчитанную быстрым(приближенным) алгоритмом

        Create fast rotation matrix
        :param epoch: date
        :return: matrix
        """

    def get_matrix_long(self, epoch: Union[Epoch, np.ndarray]) -> Matrix33:
        """

        Возвращает матрицу поворота, рассчитанную долгим(точным) по-умолчанию

        Create long rotation matrix

        :param epoch: date
        :return: matrix
        """


def build_default_rotating_cs():
    """build default rotating cs"""
    nut = Nutation1980()
    prec = PrecessionGOST()
    rot = Rotation(nut)
    return EarthRotatingCS(rot, prec, nut)
