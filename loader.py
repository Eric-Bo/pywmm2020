import numpy as np

def load_coeff():
    """
    loads the supplied coefficients into multidimensional arrays.
    returns in the order: g_t0, h_t0, g_dot, h_dot
    """
    g_t0 = np.zeros([13, 13])
    h_t0 = np.zeros([13, 13])
    g_dot = np.zeros([13, 13])
    h_dot = np.zeros([13, 13])

    coeff = np.loadtxt("WMM.COF", skiprows=1, max_rows=90)
    for row in coeff:
        n = int(row[0])
        m = int(row[1])
        g_t0[n, m] = row[2]
        h_t0[n, m] = row[3]
        g_dot[n, m] = row[4]
        h_dot[n, m] = row[5]

    return g_t0, h_t0, g_dot, h_dot