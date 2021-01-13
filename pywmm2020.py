from scipy.special import lpmn
import numpy as np
import math
from loader import load_coeff

"""
this script follows the computation outlined in the 
The US/UKWorld Magnetic Model for 2020-2025
document
"""

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
    e_squared = f * (2 - f)
    R_c = A / np.sqrt(1 - e_squared * np.sin(phi)**2)

    # equations
    p = (R_c + h) * np.cos(phi)
    z = (R_c * (1 - e_squared) + h) * np.sin(phi)
    r = np.sqrt(p**2 + z**2)
    phi_prime = np.arcsin(z / r)

    return lamb, phi_prime, r


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


def s_legendre(m, n, phi):
    """
    the Schmidt semi-normalized associated Legendre functions 
    returns the value at poit phi 
    """
    if m == 0:
        return lpmn(m, n, phi)[0][-1,-1]
    else:
        normalization = np.sqrt(2 * np.math.factorial(n - m) / 
                                np.math.factorial(n + m))
        return ((-1)**m) * normalization * lpmn(m, n, phi)[0][-1,-1]


def pnm_der(m, n, phi):
    """
    return the deivative in regard to phi of the associated legendre wih 
    sin(phi) as an argument
    """
    sum1 = (n + 1) * np.tan(phi) * s_legendre(m, n, np.sin(phi))
    sum2 = np.sqrt((n+1)**2 - m**2) * (1/np.cos(phi)) * \
                s_legendre(m, n+1, np.sin(phi))
    return sum1 - sum2


def x_prime(lamb, phi_prime, r, t):
    result = 0
    for n in range(1, N+1):
        inner_result = 0
        for m in range(n+1):
            prod1 = g(n, m, t) * np.cos(m * lamb) + h(n, m, t) * np.sin(m*lamb)
            prod2 = pnm_der(m, n, phi_prime)
            inner_result += prod1 * prod2
        result += inner_result * (a / r)**(n + 2)
    return -1 * result


def y_prime(lamb, phi_prime, r, t):
    result = 0
    for n in range(1, N+1):
        inner_result = 0
        for m in range(n+1):
            prod1 = g(n, m, t) * np.sin(m * lamb) - h(n, m, t) * np.cos(m*lamb)
            prod2 = m * s_legendre(m, n, phi_prime)
            inner_result += prod1 * prod2
        result += inner_result * (a / r)**(n + 2)
    return (1/np.cos(phi_prime)) * result


def z_prime(lamb, phi_prime, r, t):
    result = 0
    for n in range(1, N+1):
        inner_result = 0
        for m in range(n+1):
            prod1 = g(n, m, t) * np.cos(m * lamb) + h(n, m, t) * np.sin(m*lamb)
            prod2 = s_legendre(m, n, phi_prime)
            inner_result += prod1 * prod2
        result += inner_result * ((a / r)**(n + 2)) * (n + 1)
    return -1 * result


if __name__ == "__main__":
    # numerical example
    t = 2022.5
    alt_km = 100
    glat = -80
    glon = 240
    glat = glat * (np.pi/180)
    glon = glon * (np.pi/180)
    print(transform_to_spherical(glon, glat, alt_km))
    lamb, phi_prime, r = transform_to_spherical(glon, glat, alt_km)
    """
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
    """
    #print(s_legendre(1,1,phi_prime))
    print(x_prime(lamb, phi_prime, r, t))
    print(y_prime(lamb, phi_prime, r, t))
    print(z_prime(lamb, phi_prime, r, t))
