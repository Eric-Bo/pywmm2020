import scipy
import numpy as np
from loader import load_coeff

# degree of expansion of sperical harmonics
N = 12
# geomagnetic reference radius
a = 6371200
# coefficients
g_t0, h_t0, g_dot, h_dot = load_coeff()

def wmm_single(glat, glon, alt_km, year_dec):
    """
    glat: geodetic latitude in degrees
    glon: geodetic longitude in degrees
    alt_km: altitude above WGS 84 ellipsoid in km
    year_dec: decimal year
    """
    pass


def transform_to_spherical(lamb, phi, h):
    """
    transform the geodetic coordinates to spherical geocentric coordinates
    param:
    lamb: geodetic longitude in rad
    phi: geodetic latitude in rad
    h: height above WGS 84 ellipsoid

    returns:
    lamb: 
    """
    # constants from WGS 84 ellipsoid
    A = 6378137
    one_over_f = 298.257223563
    f = 1/one_over_f
    e_squared = (f) * (2 - f)
    R_c = A / np.sqrt(1 - e_squared * np.sin(phi)**2)

    # equations
    p = (R_c + h) * np.cos(phi)
    z = (R_c * (1 - e_squared) + h) * np.sin(phi)
    r = np.sqrt(p**2 + z**2)
    phi_s = np.arcsin(z / r)

    return lamb, phi_s, r


def g(n, m, t):
    t0 = 2020.0
    if t == t0:
        return g_t0[n, m]
    return g_t0[n, m] + (t - t0) * g_dot[n, m]


def h(n, m, t):
    t0 = 2020.0
    if t == t0:
        return h_t0[n, m]
    return h_t0[n, m] + (t - t0) * h_dot[n, m]


if __name__ == "__main__":
    # numerical example
    t = 2022.5
    alt_km = 100
    glat = -80
    glon = 240
    glat = glat * (np.pi/180)
    glon = glon * (np.pi/180)
    #print(transform_to_spherical(glon, glat, alt_km))
    print(g(1,0,t))
    print(g(1,1,t))
    print(g(2,0,t))
    print(g(2,1,t))
    print(g(2,2,t))
    print(h(1,0,t))
    print(h(1,1,t))
    print(h(2,0,t))
    print(h(2,1,t))
    print(h(2,2,t))