import numpy as np


def mean_anomaly_to_eccentric_anomaly(mean_anomaly, eccentricity):
    """
    Перевод из средней аномалии в эксцентрическую.
    """

    E_res = mean_anomaly + np.where(np.abs(mean_anomaly) > np.pi, -1, 1) * np.sign(mean_anomaly)
    index = E_res > E_res - 1
    while np.any(index):
        E = E_res

        E_next = E + (mean_anomaly - E + eccentricity * np.sin(E)) / (1 - eccentricity * np.cos(E))

        need_computation = np.abs(E_next - E) > 1e-8
        E_res = E_next
        index = need_computation
    return E_res


def eccentric_anomaly_to_true_anomaly(eccentric_anomaly, eccentricity):
    """Перевод из эксцентрической аномалии в истинную."""
    return np.arctan2(np.sqrt(1 - eccentricity * eccentricity) * np.sin(eccentric_anomaly),
                      np.cos(eccentric_anomaly) - eccentricity)


def mean_anomaly_to_true_anomaly_eccentric(mean_anomaly, eccentricity):
    eccentric_anomaly = mean_anomaly_to_eccentric_anomaly(mean_anomaly, eccentricity)
    true_anomaly = eccentric_anomaly_to_true_anomaly(eccentric_anomaly, eccentricity)
    return true_anomaly
