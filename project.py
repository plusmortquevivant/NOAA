from datetime import datetime, timezone

from sgp4.api import Satrec
from sgp4.api import jday
import pymap3d as pm


from coordinate_system.EarhRotating import build_default_rotating_cs
from XtoY.cartesianfromgeo import get_cartesian_geo_from_geo_angles
from coordinate_system.earth_models import get_wgs_84


#pm.geodetic2eci()
# широта и долгота ЛК
lk_coord = [55.930148, 37.518151]
print("Широта и долгота ЛК", lk_coord)

# декартовы координты ЛК
lk_cartesian = get_cartesian_geo_from_geo_angles(latitude=lk_coord[0], longitude=lk_coord[1], model=get_wgs_84())

print("Декартовы координаты ЛК", lk_cartesian)

# NOAA19 = {"name": "NOAA 19", "s": "1 33591U 09005A   21279.05525385  .00000072  00000-0  64475-4 0  9998",
#     "t": "2 33591  99.1783 302.4748 0013975 178.4339 181.6878 14.12498974652537"}
s = '1 33591U 09005A   21279.05525385  .00000072  00000-0  64475-4 0  9998'
t = '2 33591  99.1783 302.4748 0013975 178.4339 181.6878 14.12498974652537'
t0 = datetime(2021, 10, 6, 8, 6, 1, tzinfo=timezone.utc)
satellite = Satrec.twoline2rv(s, t)

jd, fr = jday(2021, 10, 6, 8, 6, 1)

# получаем векторы координат и скорости спутника в конкретный момент времени
result = satellite.sgp4(jd, fr)

print("Координаты спутника x, y, z", result[1])
print("Скорсть спутника v_x, v_y, v_z", result[2])


x, y, z = pm.eci2ecef(x=result[1][0], y=result[1][1], z=result[1][2],
                      time=t0)
print("Перевод координаты спутника НеИСО с помощью бибилиотеки", x, y, z)


# получаем матрицу перехода от ИСО в НеИСО в момент времени data
data = t0.timestamp()
rotator = build_default_rotating_cs()
rm = rotator.get_matrix(data)

# координаты в нужной системе отсчета
result_in_itrs = rm.dot(result[1])
print("Координаты спутника в НеИСО", result_in_itrs)

lk_in_itrs = rm.dot(lk_cartesian)
# print("Координаты ЛК в НеИСО в километрах", lk_in_itrs/1000)