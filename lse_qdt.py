from scipy.spatial import HalfspaceIntersection
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt


# -------------------------- Quadcoptor --------------------------
I_xx = 4.856e-3  # (kg/m^2) moment of inertia
I_yy = 4.856e-3  # (kg/m^2) moment of inertia
I_zz = 8.801e-3  # (kg/m^2) moment of inertia
m = 0.468  # (kg)    weight
Ax = 0.25  # (kg/s)  drag force coefficients
Ay = 0.25  # (kg/s)  drag force coefficients
Az = 0.25  # (kg/s)  drag force coefficients
# --------------------------- Rotor -------------------------------
l = 0.225  # (m) distance between the rotor and the center of mass
k = 2.980e-6  # lift constant of the rotor
b = 1.140e-7  # drag constant of the rotor
I_r = 3.357e-5  # (kg/m^2) moment of inertia


theta_star = np.array(
                [[1 / m, -Ax / m, -Ay / m, -Az / m, 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., (I_yy - I_zz) / I_xx, 0., 0., 1 / I_xx, 0., 0.],
                 [0., 0., 0., 0., 0., (I_zz - I_xx) / I_yy, 0., 0., 1 / I_yy, 0.],
                 [0., 0., 0., 0., 0., 0., (I_yy - I_xx) / I_zz, 0., 0., 1 / I_zz]])


def run_lst(S, Phi_S_U):

    Y: object = []
    X: object = []
    
    for t in range(len(Phi_S_U)):
        
        delta_S = S[t]
        phi_s_u = Phi_S_U[t]

        Y.append([delta_S[0], 0., 0., 0.])
        Y.append([delta_S[1], 0., 0., 0.])
        Y.append([delta_S[2], 0., 0., 0.])
        Y.append([0., delta_S[3], delta_S[4], delta_S[5]])

        X.append([phi_s_u[0], phi_s_u[3], 0., 0., 0., 0., 0., 0., 0., 0.])
        X.append([phi_s_u[1], 0., phi_s_u[4], 0., 0., 0., 0., 0., 0., 0.])
        X.append([phi_s_u[2], 0., 0., phi_s_u[5], 0., 0., 0., 0., 0., 0.])
        X.append([0., 0., 0., 0., phi_s_u[6], phi_s_u[7], phi_s_u[8], phi_s_u[9], phi_s_u[10], phi_s_u[11]])

    YY = np.array(Y)
    XX = np.array(X)

    theta_hat = (sc.linalg.pinv(XX) @ YY).T
    theta_hat_ = np.array([theta_hat[0, 0], theta_hat[0, 1], theta_hat[0, 2], theta_hat[0, 3],
                          theta_hat[1, 4], theta_hat[1, 7],
                          theta_hat[2, 5], theta_hat[2, 8],
                          theta_hat[3, 6], theta_hat[3, 9]])
    return theta_hat_
