import numpy as np
from numpy import arange, array, polyder, polyval
from numpy.linalg import inv
from scipy.linalg import expm, logm


def straight_trajectory(T_start, T_end, t_start, t_end, h):
    """
    Create a straight line  trajectory in SE(3) between given start and end
    poses

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
    # s(t) = a[0] * t**3 + a[1] * t**2 + a[2] * t + a[3]
    a = (inv(A) @ b)[::-1]
    
    # Time derivative of the time scaling polynomial
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


def circular_trajectory(T_start, r, f, t_start, t_end, h, plane='xy'):
    x_start, y_start, z_start = T_start[0:3, 3]

    # Time vector for the trajectory
    t = arange(t_start, t_end + h, h)

    if plane == 'xy' or plane == 'yx':
        x = x_start - r * (1 - np.cos(2 * np.pi * f * t))
        y = y_start - r * np.sin(2 * np.pi * f * t)
        z = np.ones(t.shape) * z_start
    elif plane == 'yz' or plane == 'zy':
        x = np.ones(t.shape) * x_start
        y = y_start - r * np.sin(2 * np.pi * f * t)
        z = z_start - r * (1 - np.cos(2 * np.pi * f * t))
    elif plane == 'xz' or plane == 'zx':
        x = x_start - r * (1 - np.cos(2 * np.pi * f * t))
        y = np.ones(t.shape) * y_start
        z = z_start - r * np.sin(2 * np.pi * f * t)

    return np.vstack((x, y, z)).transpose()


def eight_trajectory(T_start, r, f, t_start, t_end, h, plane='xy'):
    x_start, y_start, z_start = T_start[0:3, 3]

    # Time vector for the trajectory
    t = arange(t_start, t_end + h, h)

    if plane == 'xy' or plane == 'yx':
        x = x_start - r * np.sin(2 * np.pi * f * t)
        y = y_start - r * np.sin(2 * np.pi * f * t) * np.cos(2 * np.pi * f * t)
        z = np.ones(t.shape) * z_start
    # elif plane == 'yz' or plane == 'zy':
    #     x = np.ones(t.shape) * x_start
    #     y = y_start - r * np.sin(2 * np.pi * f * t)
    #     z = z_start - r * (1 - np.cos(2 * np.pi * f * t))
    # elif plane == 'xz' or plane == 'zx':
    #     x = x_start - r * (1 - np.cos(2 * np.pi * f * t))
    #     y = np.ones(t.shape) * y_start
    #     z = z_start - r * np.sin(2 * np.pi * f * t)

    return np.vstack((x, y, z)).transpose()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Initial pose on the trajectory
    T0 = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])

    # Final pose on the trajectory
    T1 = np.array([[0.7071069, -0.7071066, 0.0, 1.0],
                   [0.7071066, 0.7071069, 0.0, 2.0],
                   [0.0, 0.0, 1.0, 3.0],
                   [0.0, 0.0, 0.0, 1.0]])

    # Initial and the final time on the trajectory
    t0, t1 = 0.0, 8.0

    # Time step for the trajectory
    step = 1e-3

    # Time vector
    t_vec = arange(t0, t1 + step, step)

    # Straight trajectory
    tr, trp = straight_trajectory(T0, T1, t0, t1, step)

    # Plot the trajectory
    fig, ax = plt.subplots()
    ax.plot(t_vec, tr)
    # Circular trajectory
    tr = circular_trajectory(T0, 0.25, 1.0, t0, t1, step, plane='xy')

    # Eight trajectory
    tr = eight_trajectory(T0, 0.25, 1.0, t0, t1, step, plane='xy')
