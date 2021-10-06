"""
Модели Земли
"""
from dataclasses import dataclass


@dataclass
class EarthEllipsoidModel:
    """
    Класс моделей Земли
    """
    eq_rad: float  # Экваториальный радиус
    pol_rad: float  # полярный радиус


def get_spherical_model() -> EarthEllipsoidModel:
    """
    Возвращает сферическую модель земли
    """
    average_rad = 6371.302e3  # Средний радиус Земли
    return EarthEllipsoidModel(average_rad, average_rad)


def get_wgs_84() -> EarthEllipsoidModel:
    """
    Возвращает модель WGS84
    """
    eq_rad = 6378137
    pol_rad = 6356752.3142
    return EarthEllipsoidModel(eq_rad, pol_rad)
