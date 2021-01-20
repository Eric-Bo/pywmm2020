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
    glat = glat * np.pi/180
    glon = glon * np.pi/180
    t = year_dec
    h = alt_km * 1000
    lamb, phi_prime, r = transform_to_spherical(glon, glat, alt_km*1000)

    x_prime = x_prime(lamb, phi_prime, r, t)
    y_prime = y_prime(lamb, phi_prime, r, t)
    z_prime = z_prime(lamb, phi_prime, r, t)

    x_dot_prime = x_dot_prime(lamb, phi_prime, r, t)
    y_dot_prime = y_dot_prime(lamb, phi_prime, r, t)
    z_dot_prime = z_dot_prime(lamb, phi_prime, r, t)

    x = x(x_prime, z_prime, phi_prime, phi)
    y = y(y_prime)
    z = z(x_prime, z_prime, phi_prime, phi)

    x_dot = x_dot(x_prime, z_prime, phi_prime, phi)
    y_dot = y_dot(y_dot_prime)
    z_dot = z_dot(x_prime, z_prime, phi_prime, phi)

    H = H(x, y)
    f = f(h, z)
    i = i(z, h)
    d = d(y, x)

    h_dot = h_dot(x, x_dot, y, y_dot, H)
    f_dot = f_dot(x, x_dot, y, y_dot, z, z_dot, f)
    i_dot = i_dot(z, z_dot, h, h_dot, f)
    d_dot = d_dot(x, x_dot, y, y_dot, h)


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
    R_c = A / np.sqrt(1 - e_squared * (np.sin(phi)**2))

    # equations
    p = (R_c + h) * np.cos(phi)
    z = (R_c * (1 - e_squared) + h) * np.sin(phi)
    r = np.sqrt(p**2 + z**2)
    phi_prime = np.arcsin(z / r)

    return lamb, phi_prime, r


def g(n, m, t):
    t0 = 2020.0
    return g_t0[n, m] + (t - t0) * g_dot[n, m]


def h(n, m, t):
    t0 = 2020.0
    return h_t0[n, m] + (t - t0) * h_dot[n, m]


def s_legendre(m, n, phi_prime):
    """
    the Schmidt semi-normalized associated Legendre functions 
    returns the value at poit phi 
    """
    if m == 0:
        return lpmn(m, n, phi_prime)[0][-1,-1]
    else:
        normalization = np.sqrt(2 * np.math.factorial(n - m) / 
                                np.math.factorial(n + m))
        return ((-1)**m) * normalization * lpmn(m, n, phi_prime)[0][-1,-1]


def pnm_der(m, n, phi_prime):
    """
    return the deivative in regard to phi of the associated legendre wih 
    sin(phi) as an argument
    """
    sum1 = (n + 1) * np.tan(phi_prime) * s_legendre(m, n, np.sin(phi_prime))
    sum2 = np.sqrt((n+1)**2 - m**2) * (1/np.cos(phi_prime)) * \
                s_legendre(m, n+1, np.sin(phi_prime))
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
            prod2 = m * s_legendre(m, n, np.sin(phi_prime))
            inner_result += prod1 * prod2
        result += inner_result * (a / r)**(n + 2)
    return (1/np.cos(phi_prime)) * result


def z_prime(lamb, phi_prime, r, t):
    result = 0
    for n in range(1, N+1):
        inner_result = 0
        for m in range(n+1):
            prod1 = g(n, m, t) * np.cos(m * lamb) + h(n, m, t) * np.sin(m*lamb)
            prod2 = s_legendre(m, n, np.sin(phi_prime))
            inner_result += prod1 * prod2
        result += inner_result * ((a / r)**(n + 2)) * (n + 1)
    return -1 * result


def x_dot_prime(lamb, phi_prime, r, t):
    result = 0
    for n in range(1, N+1):
        inner_result = 0
        for m in range(n+1):
            prod1 = g_dot[n, m] * np.cos(m * lamb) + h_dot[n, m] * np.sin(m*lamb)
            prod2 = pnm_der(m, n, phi_prime)
            inner_result += prod1 * prod2
        result += inner_result * (a / r)**(n + 2)
    return -1 * result


def y_dot_prime(lamb, phi_prime, r, t):
    result = 0
    for n in range(1, N+1):
        inner_result = 0
        for m in range(n+1):
            prod1 = g_dot[n, m] * np.sin(m * lamb) - h_dot[n, m] * np.cos(m*lamb)
            prod2 = m * s_legendre(m, n, np.sin(phi_prime))
            inner_result += prod1 * prod2
        result += inner_result * (a / r)**(n + 2)
    return (1/np.cos(phi_prime)) * result


def z_dot_prime(lamb, phi_prime, r, t):
    result = 0
    for n in range(1, N+1):
        inner_result = 0
        for m in range(n+1):
            prod1 = g_dot[n, m] * np.cos(m * lamb) + h_dot[n, m] * np.sin(m*lamb)
            prod2 = s_legendre(m, n, np.sin(phi_prime))
            inner_result += prod1 * prod2
        result += inner_result * ((a / r)**(n + 2)) * (n + 1)
    return -1 * result


def x(x_prime, z_prime, phi_prime, phi):
    return x_prime * np.cos(phi_prime - phi) - z_prime * np.sin(phi_prime - phi)


def y(y_prime):
    return y_prime


def z(x_prime, z_prime, phi_prime, phi):
    return x_prime * np.sin(phi_prime - phi) + z_prime * np.cos(phi_prime - phi)


def x_dot(x_prime, z_prime, phi_prime, phi):
    return x_dot_prime * np.cos(phi_prime - phi) - z_dot_prime * np.sin(phi_prime - phi)


def y_dot(y_dot_prime):
    return y_dot_prime


def z_dot(x_prime, z_prime, phi_prime, phi):
    return x_dot_prime * np.sin(phi_prime - phi) + z_dot_prime * np.cos(phi_prime - phi)


def H(x, y):
    return np.sqrt(x**2 + y**2)


def f(h, z):
    return np.sqrt(h**2 + z**2)


def i(z, h):
    if h != 0:
        return np.arctan(z / h)
    else:
        return np.arctan(z / 1e-9)


def d(y, x):
    if x != 0:
        return np.arctan(y / x)
    else:
        return np.arctan(y / 1e-9)


def h_dot(x, x_dot, y, y_dot, h):
    return (x * x_dot + y * y_dot) / h


def f_dot(x, x_dot, y, y_dot, z, z_dot, f):
    return (x * x_dot + y * y_dot + z * z_dot) / f


def i_dot(z, z_dot, h, h_dot, f):
    return (h * z_dot - z * h_dot) / f**2


def d_dot(x, x_dot, y, y_dot, h):
    return (x * y_dot - y * x_dot) / h**2


if __name__ == "__main__":
    # numerical example
    t = 2022.5
    alt_km = 100
    glat = -80
    glon = 240
    glat = glat * (np.pi/180)
    glon = glon * (np.pi/180)
    phi = glat
    print(phi)
    print(transform_to_spherical(glon, glat, alt_km *1000))
    lamb, phi_prime, r = transform_to_spherical(glon, glat, alt_km*1000)
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

    x_prime = x_prime(lamb, phi_prime, r, t)
    y_prime = y_prime(lamb, phi_prime, r, t)
    z_prime = z_prime(lamb, phi_prime, r, t)
    print("x_prime: " + str(x_prime))
    print("y_prime: " + str(y_prime))
    print("z_prime: " + str(z_prime))

    x_dot_prime = x_dot_prime(lamb, phi_prime, r, t)
    y_dot_prime = y_dot_prime(lamb, phi_prime, r, t)
    z_dot_prime = z_dot_prime(lamb, phi_prime, r, t)
    print("x_dot_prime: " + str(x_dot_prime))
    print("y_dot_prime: " + str(y_dot_prime))
    print("z_dot_prime: " + str(z_dot_prime))

    x = x(x_prime, z_prime, phi_prime, phi)
    y = y(y_prime)
    z = z(x_prime, z_prime, phi_prime, phi)
    print("x: " + str(x))
    print("y: " + str(y))
    print("z: " + str(z))

    x_dot = x_dot(x_prime, z_prime, phi_prime, phi)
    y_dot = y_dot(y_dot_prime)
    z_dot = z_dot(x_prime, z_prime, phi_prime, phi)
    print("x_dot: " + str(x_dot))
    print("y_dot: " + str(y_dot))
    print("z_dot: " + str(z_dot))

    h = H(x, y)
    f = f(h, z)
    i = i(z, h)
    d = d(y, x)
    print("F: " + str(f))
    print("H: " + str(h))
    print("D: " + str(d))
    print("I: " + str(i))


