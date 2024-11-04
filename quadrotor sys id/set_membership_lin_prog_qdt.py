from scipy.spatial import HalfspaceIntersection
from cvxopt import matrix, solvers
from scipy.spatial import ConvexHull, convex_hull_plot_2d
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

ground_truth = [1 / m,
                -Ax / m, -Ay / m, -Az / m,
                1 * (I_yy - I_zz) / I_xx, 1 / I_xx,
                1 * (I_zz - I_xx) / I_yy, 1 / I_yy,
                (I_xx - I_yy) / I_zz, 1 / I_zz]

def run_set_membership(Delta_S, Phi_S_U, w_max):

    # ----------------------finding a feasible point----------------------------
    AA = []
    bb = []
    cc = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    Ab = []

    for t in range(len(Phi_S_U)):

        delta_S = Delta_S[t]
        phi_s_u = Phi_S_U[t]

        # --------------------------------------------- Half Spaces ----------------------------------------------------
        AA.append([phi_s_u[0], phi_s_u[3], 0., 0., 0., 0., 0., 0., 0., 0.])
        bb.append((w_max + delta_S[0]))
        AA.append([-phi_s_u[0], -phi_s_u[3], 0., 0., 0., 0., 0., 0., 0., 0.])
        bb.append((w_max - delta_S[0]))

        AA.append([phi_s_u[1], 0., phi_s_u[4], 0., 0., 0., 0., 0., 0., 0.])
        bb.append((w_max + delta_S[1]))
        AA.append([-phi_s_u[1], 0., -phi_s_u[4], 0., 0., 0., 0., 0., 0., 0.])
        bb.append((w_max - delta_S[1]))

        AA.append([phi_s_u[2], 0., 0., phi_s_u[5], 0., 0., 0., 0., 0., 0.])
        bb.append((w_max + delta_S[2]))
        AA.append([-phi_s_u[2], 0., 0., -phi_s_u[5], 0., 0., 0., 0., 0., 0.])
        bb.append((w_max - delta_S[2]))

        AA.append([0., 0., 0., 0., phi_s_u[6], phi_s_u[9], 0., 0., 0., 0.])
        bb.append((w_max + delta_S[3]))
        AA.append([0., 0., 0., 0., - phi_s_u[6], - phi_s_u[9], 0., 0., 0., 0.])
        bb.append((w_max - delta_S[3]))

        AA.append([0., 0., 0., 0., 0., 0., phi_s_u[7], phi_s_u[10], 0., 0.])
        bb.append((w_max + delta_S[4]))
        AA.append([0., 0., 0., 0., 0., 0., - phi_s_u[7], - phi_s_u[10], 0., 0.])
        bb.append((w_max - delta_S[4]))

        AA.append([0., 0., 0., 0., 0., 0., 0., 0., phi_s_u[8], phi_s_u[11]])
        bb.append((w_max + delta_S[5]))
        AA.append([0., 0., 0., 0., 0., 0., 0., 0., -phi_s_u[8], -phi_s_u[11]])
        bb.append((w_max - delta_S[5]))
        # --------------------------------------------------------------------------------------------------------------

        Ab.append([phi_s_u[0], phi_s_u[3], 0., 0., 0., 0., 0., 0., 0., 0., - (w_max + delta_S[0])])
        Ab.append([-phi_s_u[0], -phi_s_u[3], 0., 0., 0., 0., 0., 0., 0., 0., - (w_max - delta_S[0])])

        Ab.append([phi_s_u[1], 0., phi_s_u[4], 0., 0., 0., 0., 0., 0., 0., - (w_max + delta_S[1])])
        Ab.append([-phi_s_u[1], 0., -phi_s_u[4], 0., 0., 0., 0., 0., 0., 0., - (w_max - delta_S[1])])

        Ab.append([phi_s_u[2], 0., 0., phi_s_u[5], 0., 0., 0., 0., 0., 0., - (w_max + delta_S[2])])
        Ab.append([-phi_s_u[2], 0., 0., -phi_s_u[5], 0., 0., 0., 0., 0., 0., - (w_max - delta_S[2])])

        Ab.append([0., 0., 0., 0., phi_s_u[6], phi_s_u[9], 0., 0., 0., 0., - (w_max + delta_S[3])])
        Ab.append([0., 0., 0., 0., -phi_s_u[6], -phi_s_u[9], 0., 0., 0., 0., - (w_max - delta_S[3])])

        Ab.append([0., 0., 0., 0., 0., 0., phi_s_u[7], phi_s_u[10], 0., 0., - (w_max + delta_S[4])])
        Ab.append([0., 0., 0., 0., 0., 0., -phi_s_u[7], -phi_s_u[10], 0., 0., - (w_max - delta_S[4])])

        Ab.append([0., 0., 0., 0., 0., 0., 0., 0., phi_s_u[8], phi_s_u[11], - (w_max + delta_S[5])])
        Ab.append([0., 0., 0., 0., 0., 0., 0., 0., -phi_s_u[8], -phi_s_u[11], - (w_max - delta_S[5])])

    print("---------------------- finding a feasible point by cvxopt -- T = ", len(Phi_S_U), " -----------")
    sol = solvers.lp(matrix(cc), matrix(AA).trans(), matrix(bb))

    # ----------------------------------------------------  half_space intersection ------ -------------------------------------
    if sol['status'] == 'optimal':
      # print("feasible point:", np.array(sol['x']).reshape(12, ))
      feasible_point = np.array(sol['x']).reshape(10, )
      half_spaces = np.array(Ab, dtype=object)
      hs = HalfspaceIntersection(half_spaces, feasible_point)
      return hs.intersections, sol['status']
    else:
      print("continiue with ground truth")
      feasible_point = np.array(ground_truth)
      half_spaces = np.array(Ab, dtype=object)
      hs = HalfspaceIntersection(half_spaces, feasible_point)
      return hs.intersections, sol['status']