from sgp4.api import Satrec

s = '1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991'
t = '2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482'
satellite = Satrec.twoline2rv(s, t)

semimajor = satellite.a * satellite.radiusearthkm
eccentricity = satellite.ecco
inclination = satellite.inclo
argument_periapsis = satellite.argpo
ascending_node = satellite.Om
mean_anomaly = satellite.mo
gravitational_parameter = satellite.mu
