import numpy as np
import math


def xyz_to_rfilam(x, y, z):  # pylint: disable=invalid-name
    """
    Converts a Cartesian coordinate system to a spherical coordinate system
    :param x: X
    :param y: Y
    :param z: Z
    :return: coordinates in spherical coordinate system
    """
    r = np.linalg.norm(np.array((x, y, z)))
    fi = math.asin(z / r)
    if x == 0:
        if y > 0:
            lam = np.pi / 2
        else:
            lam = -np.pi / 2
    else:
        lam = np.arctan(y/x)
        if x < 0:
            lam += np.pi
    return r, fi, lam
