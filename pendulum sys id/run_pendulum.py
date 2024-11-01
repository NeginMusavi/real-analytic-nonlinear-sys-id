from scipy.spatial import HalfspaceIntersection
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba
# import seaborn as sns

from quadrotor_dynamics import QuadrotorDynamics, euler_to_quaternion

if __name__ == "__main__":
    time_hor = [1000]
    disturbance: str = "trunc_guass"
    input: str = "trunc_guass"
    w_max = 1

    intersection_points = []

    for k in range(len(time_hor)):
        q0, q1, q2, q3 = euler_to_quaternion(10 * np.pi/180, 10 * np.pi/180, 10 * np.pi/180)
        x0 = [0., 0., 1., 0., 0., 0., q0, q1, q2, q3, 0., 0., 0.]
        qdt = QuadrotorDynamics(input, disturbance)
        qdt.get_trajectory_3(x0, time_hor[k])
        qdt.plot_trajectory()
        print(qdt.phi_s_u_list[10])
        print(qdt.b_s_list[10])







