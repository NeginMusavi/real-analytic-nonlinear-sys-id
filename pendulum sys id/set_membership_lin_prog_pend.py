from scipy.spatial import HalfspaceIntersection
from cvxopt import matrix, solvers
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt

m = 0.1
l = 0.5
ground_truth = [1 / l, 1 / m / l / l]

def run_set_membership(Delta_S, Phi_S_U, w_max):

    # ----------------------finding a feasible point----------------------------
    AA = []
    bb = []
    cc = [0., 0.]
    Ab = []

    for t in range(len(Phi_S_U)):

        delta_S = Delta_S[t]
        phi_s_u = Phi_S_U[t]

        # --------------------------------------------- Half Spaces ----------------------------------------------------
        AA.append([phi_s_u[0], phi_s_u[1]])
        bb.append((w_max + delta_S))
        AA.append([-phi_s_u[0], -phi_s_u[1]])
        bb.append((w_max - delta_S))

        # --------------------------------------------------------------------------------------------------------------

        Ab.append([phi_s_u[0], phi_s_u[1], - (w_max + delta_S)])
        Ab.append([-phi_s_u[0], -phi_s_u[1], - (w_max - delta_S)])


    print("---------------------- finding a feasible point by cvxopt -- T = ", len(Phi_S_U), " -----------")
    sol = solvers.lp(matrix(cc), matrix(AA).trans(), matrix(bb))

    # ----------------------------------------------------  half_space intersection ------ -------------------------------------
    if sol['status'] == 'optimal':
      # print("feasible point:", np.array(sol['x']).reshape(12, ))
      feasible_point = np.array(sol['x']).reshape(2, )
      half_spaces = np.array(Ab, dtype=object)
      hs = HalfspaceIntersection(half_spaces, feasible_point)
      return hs.intersections, sol['status']
    else:
      print("continiue with ground truth")
      feasible_point = np.array(ground_truth)
      half_spaces = np.array(Ab, dtype=object)
      hs = HalfspaceIntersection(half_spaces, feasible_point)
      return hs.intersections, sol['status']