import numpy as np
import math as math


def oetoeci(mu, semimajor, eccentricity, inclination, argument_periapsis, ascending_node, true_anomaly):
    r = np.zeros(3)
    v = np.zeros(3)

    slr = semimajor * (1 - eccentricity * eccentricity)
    rm = slr / (1 + eccentricity * math.cos(true_anomaly))

    arglat = argument_periapsis + true_anomaly
    sarglat = math.sin(arglat)
    carglat = math.cos(arglat)

    c4 = math.sqrt(mu / slr)
    c5 = eccentricity * math.cos(argument_periapsis) + carglat
    c6 = eccentricity * math.sin(argument_periapsis) + sarglat
    sinc = math.sin(inclination)
    cinc = math.cos(inclination)
    sraan = math.sin(ascending_node)
    craan = math.cos(ascending_node)

    r[0] = rm * (craan * carglat - sraan * cinc * sarglat)
    r[1] = rm * (sraan * carglat + cinc * sarglat * craan)
    r[2] = rm * sinc * sarglat

    v[0] = -c4 * (craan * c6 + sraan * cinc * c5)
    v[1] = -c4 * (sraan * c6 - craan * cinc * c5)
    v[2] = c4 * c5 * sinc

    return r, v
