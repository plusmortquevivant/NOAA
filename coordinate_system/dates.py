"""Module convert date"""
import datetime
import math
from typing import Union, Sequence
import numpy as np
from .angles import degrees_to_rad, degrees_to_rad_multi

from .types import Epoch, EpochMulti

SECONDS_PER_DAY = 24 * 3600

JULIAN_CENTURY = 36525
J2000_ERA_BEGIN = datetime.datetime(year=2000, month=1, day=1, hour=12, minute=0, second=0, tzinfo=datetime.timezone.utc)
J1986_ERA_BEGIN = datetime.datetime(year=1986, month=1, day=1, hour=12, minute=0, second=0, tzinfo=datetime.timezone.utc)


def epoch_to_jdn(epoch: Union[Epoch, np.ndarray]) -> Union[int, Sequence[int]]:
    """
    Convert Epoch to julian day number(JDN)
    (see https://ru.wikipedia.org/wiki/%D0%AE%D0%BB%D0%B8%D0%B0%D0%BD%D1%81%D0%BA%D0%B0%D1%8F_%D0%B4%D0%B0%D1%82%D0%B0)  # pylint: disable=line-too-long # noqa: E501,W505

    :param epoch: time to convert
    :return: day number
    """
    t = datetime.datetime.utcfromtimestamp(epoch)  # pylint: disable=invalid-name
    a = (14 - t.month) // 12  # pylint: disable=invalid-name
    y = t.year + 4800 - a  # pylint: disable=invalid-name
    m = t.month + 12 * a - 3  # pylint: disable=invalid-name
    res = t.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return res


def epoch_to_jd(epoch: Union[Epoch, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert Epoch to julian date (JD)
    :param epoch: time to convert
    :return: julian date
    """
    t = datetime.datetime.utcfromtimestamp(epoch)  # pylint: disable=invalid-name
    res = epoch_to_jdn(epoch) + (t.hour - 12) / 24 + t.minute / 1440 + t.second / 86400
    return res


def epoch_to_jdn_multi(epochs: EpochMulti) -> np.ndarray:
    """
    Convert Epoch to julian day number(JDN)
    (see https://ru.wikipedia.org/wiki/%D0%AE%D0%BB%D0%B8%D0%B0%D0%BD%D1%81%D0%BA%D0%B0%D1%8F_%D0%B4%D0%B0%D1%82%D0%B0)  # pylint: disable=line-too-long # noqa: E501,W505

    :param epoch: time to convert
    :return: day number
    """
    t = epochs.astype('datetime64[s]')  # pylint: disable=invalid-name
    month = (t.astype('datetime64[M]').astype(int) % 12 + 1).astype(float)
    year = (t.astype('datetime64[Y]').astype(int) + 1970).astype(float)
    day = (t.astype('datetime64[D]') - t.astype('datetime64[M]') + 1).astype(float)
    a = (14 - month) // 12  # pylint: disable=invalid-name
    y = year + 4800 - a  # pylint: disable=invalid-name
    m = month + 12 * a - 3  # pylint: disable=invalid-name
    return day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045


def epoch_to_jd_multi(epochs: EpochMulti) -> np.ndarray:
    """
    Convert Epoch to julian date (JD)
    :param epoch: time to convert
    :return: array julian date
    """
    t = epochs.astype('datetime64[s]')  # pylint: disable=invalid-name
    hour = (t.astype('datetime64[h]') - t.astype('datetime64[D]')).astype(float)
    minute = (t.astype('datetime64[m]') - t.astype('datetime64[h]')).astype(float)
    second = (t.astype('datetime64[s]') - t.astype('datetime64[m]')).astype(float)
    return epoch_to_jdn_multi(epochs) + (hour - 12) / 24 + minute / 1440 + second / 86400


# J2000_ERA_BEGIN_JD = epoch_to_jd(J2000_ERA_BEGIN.timestamp())  # съедала 29 секунд
J2000_ERA_BEGIN_JD = 2451545.


def number_of_leap_seconds(epoch: Epoch):  # pylint: disable=unused-argument
    """ Return number of leap second"""
    return 32.0


def utc2ut1(epoch: Union[Epoch, np.ndarray]) -> Union[Epoch, np.ndarray]:
    """Convert UTC to UT1"""
    return epoch


def epoch_to_gmst(epoch: Union[Epoch, np.ndarray]) -> Union[Epoch, np.ndarray]:
    """
    Convert Epoch to Greenwich Mean Sidereal Time
    :param epoch: time to convert
    :return: GMST in radians
    """
    epoch = utc2ut1(epoch)
    if isinstance(epoch, Epoch):
        JD = epoch_to_jd(epoch)
    else:
        JD = epoch_to_jd_multi(epoch)
    D = JD - J2000_ERA_BEGIN_JD  # pylint: disable=invalid-name
    T = D / JULIAN_CENTURY  # pylint: disable=invalid-name
    gmst_in_deg = 280.46061837 + 360.98564736629 * D + 0.000388 * T ** 2
    if isinstance(epoch, Epoch):
        gmst_in_deg = gmst_in_deg - int(gmst_in_deg / 360) * 360
    else:
        gmst_in_deg = gmst_in_deg - (gmst_in_deg / 360).astype(int) * 360
    return degrees_to_rad(gmst_in_deg)


def epoch_to_gmst2(epoch: Epoch) -> float:
    """
    Convert Epoch to Greenwich Mean Sidereal Time
    :param epoch: time to convert
    :return: GMST in radians
    """
    epoch = utc2ut1(epoch)
    JD = epoch_to_jd(epoch)  # pylint: disable=invalid-name
    D = JD - J2000_ERA_BEGIN_JD  # pylint: disable=invalid-name
    T = D / JULIAN_CENTURY  # pylint: disable=invalid-name
    gmst_in_sec = 24110.54841 + 8640184.812866 * T + 0.093104 * T ** 2 - 0.0000062 * T ** 3
    gmst_in_deg = gmst_in_sec * 360 / 86400 + 180 + 360 * D
    gmst_in_deg = gmst_in_deg - int(gmst_in_deg / 360) * 360
    return degrees_to_rad(gmst_in_deg)


def epoch_to_gmst3(epoch: Epoch) -> float:
    """
    Convert Epoch to Greenwich Mean Sidereal Time
    :param epoch: time to convert
    :return: GMST in radians
    """
    epoch = utc2ut1(epoch)
    t = datetime.datetime.fromtimestamp(epoch)  # pylint: disable=invalid-name
    t0 = t.replace(hour=0, minute=0, second=0, microsecond=0)  # pylint: disable=invalid-name
    JD = epoch_to_jd(epoch)  # pylint: disable=invalid-name
    JD0 = epoch_to_jd(t0.timestamp())  # pylint: disable=invalid-name
    D = JD - J2000_ERA_BEGIN_JD  # pylint: disable=invalid-name
    D0 = JD0 - J2000_ERA_BEGIN_JD  # pylint: disable=invalid-name
    H = (JD - JD0) * 24  # pylint: disable=invalid-name
    T = D / JULIAN_CENTURY  # pylint: disable=invalid-name

    gmst_in_hour = 6.697374558 + 0.06570982441908 * D0 + 1.00273790935 * H + 0.000026 * T ** 2
    gmst_in_hour = gmst_in_hour - int(gmst_in_hour / 24) * 24
    return gmst_in_hour / 24 * 2 * np.pi


def utc2tdt(utc: Epoch) -> float:
    """Convert UTC to TDT"""
    return utc + 32.184 + number_of_leap_seconds(utc)


utc2tdt_multi = utc2tdt  # pylint: disable=invalid-name


def utc2tdb(epoch: Epoch) -> float:
    """Convert UTC to TDB"""
    tt = utc2tdt(epoch)  # pylint: disable=invalid-name
    jd = epoch_to_jd(epoch)  # pylint: disable=invalid-name

    g = degrees_to_rad(357.53 + 0.9856003 * (jd - 2451545.0))  # pylint: disable=invalid-name
    return tt + 0.001658 * math.sin(g) + 0.000014 * math.sin(2 * g)


def utc2tdb_multi(epochs: EpochMulti) -> np.ndarray:
    """Convert UTC to TDB"""
    tt = utc2tdt_multi(epochs)  # pylint: disable=invalid-name
    jd = epoch_to_jd_multi(epochs)  # pylint: disable=invalid-name

    g = degrees_to_rad_multi(357.53 + 0.9856003 * (jd - 2451545.0))  # pylint: disable=invalid-name
    return tt + 0.001658 * np.sin(g) + 0.000014 * np.sin(2 * g)


def days_from_beginning_of_year(epoch: Epoch) -> int:
    """
    Return number of days since the beginning of the year
    :param epoch: Epoch
    :return: number of days
    """
    cur_date = datetime.datetime.fromtimestamp(epoch)
    new_year_date = cur_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return (cur_date - new_year_date).days
