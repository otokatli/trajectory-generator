from numpy import arange, array, polyder, polyval
from numpy.linalg import inv
from scipy.linalg import expm, logm


def geodesic(T_start, T_end, t_start, t_end, h):
    """
    Create a geodesic trajectory (in SE(3)) between given start and end poses

    :param T_start: Initial pose on the trajectory
    :param T_end: Final pose on the trajectory
    :param t_start: Initial time
    :param t_end: Final time
    :param h: Time step
    :return: homogeneous coordinates on the trajectory at each time step
    """

    A = array([[1.0, t_start, t_start ** 2, t_start ** 3],
               [0.0, 1.0, 2.0 * t_start, 3.0 * t_start ** 2],
               [1.0, t_end, t_end ** 2, t_end ** 3],
               [0.0, 1.0, 2.0 * t_end, 3.0 * t_end ** 2]])

    b = array([0.0, 0.0, 1.0, 0.0])

    # coefficients of the cubic time scaling polynomial
    # s(t) = a[0] * t**3 + a[1] * t**2 + a[2] * t + a[3] * t
    a = (inv(A) @ b)[::-1]
    
    # Time derivative of the time scaling polynimial
    ap = polyder(a)

    # Time vector for the trajectory
    t = arange(t_start, t_end + h, h)

    s_values = polyval(a, t)
    sp_values = polyval(ap, t)

    return ([T_start @ expm(logm(inv(T_start) @ T_end) * s) for s in
            s_values],
            [T_start @ logm(inv(T_start) @ T_end) @ expm(logm(inv(T_start) 
                                                              @ T_end) * s)
             * sp for (s, sp) in zip(s_values, sp_values)])
