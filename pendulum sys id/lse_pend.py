from scipy.spatial import HalfspaceIntersection
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt


# -------------------------- Pendulum --------------------------
m = 0.5  # (kg)    mass
l = 0.1  # (m) rod length
b = 0.1  #


theta_star = np.array([- b / (m * l * l), - 1 / l, 1 / (m * l * l)])


def run_lst(S, Phi_S_U):

    Y: object = []
    X: object = []
    
    for t in range(len(Phi_S_U)):
        
        delta_S = S[t]
        phi_s_u = Phi_S_U[t]

        Y.append([delta_S])
        X.append([phi_s_u[0], phi_s_u[1], phi_s_u[2]])


    YY = np.array(Y)
    XX = np.array(X)

    theta_hat = (sc.linalg.pinv(XX) @ YY).T
    theta_hat_ = np.array([theta_hat[0,0], theta_hat[0,1], theta_hat[0,2]])
    return theta_hat_
