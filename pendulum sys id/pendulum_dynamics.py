import numpy as np
from scipy.stats import truncnorm
import random
import matplotlib.pyplot as plt
import math


g = 9.81  # (m/s^2)  gravity constant
dt = 0.001  # time_step for discrete-time system


def system_parameters():
    m = 0.468  # (kg)    mass
    l = 0.225  # (m) distance between the rotor and the center of mass
    b = 1.140e-7  # drag constant of the rotor
    return m, l, b

def generate_u(input_, time_hor, s_, mean, std, u_max, lb, ub):  # noise in control input
    if input_ == "trunc_guass":
        np.random.seed(s_)
        rv = truncnorm(-u_max, u_max, loc=mean, scale=std)
        r1 = rv.rvs(size=time_hor)
        return r1
    elif input_ == "uniform":
        np.random.seed(s_)
        r1 = np.random.uniform(low=lb, high=ub, size=time_hor)
        return r1

def generate_w(distr, time_hor, s_, mean, std, w_max, lb, ub):  # disturbance
    if distr == "trunc_guass":
        np.random.seed(s_)
        rv = truncnorm(-w_max, w_max, loc=mean, scale=std)
        r1 = rv.rvs(size=time_hor)
        rv = truncnorm(-w_max, w_max, loc=mean, scale=std)
        r2 = rv.rvs(size=time_hor)
        return r1, r2
    elif distr == "uniform":
        np.random.seed(s_)
        r1 = np.random.uniform(low=lb, high=ub, size=time_hor)
        r2 = np.random.uniform(low=lb, high=ub, size=time_hor)
        return r1, r2


class SimplePendulumDynamics:
    def __init__(self, distr, input):
        self.distr = distr
        self.input = input
        self.m, self.l, self.b = system_parameters()
        self.alpha_list = []
        self.omega_list = []
        self.phi_s_u_list = []
        self.b_s_list = []

    def plot_trajectory(self):
        t_list = np.array(range(len(self.px_list))) * dt
        fig = plt.figure()
        plt.plot(t_list, self.alpha_list, label='$\alpha$')
        # plt.title("Quadrotor's Position")
        plt.xlabel('time (s)')
        plt.ylabel('$\alpha$')
        plt.legend()

        fig = plt.figure()
        plt.plot(t_list, self.omega_list, label='$\omega$')
        # plt.title("Quadrotor's Angular Velocity")
        plt.xlabel('time (s)')
        plt.ylabel('$\omega$')
        plt.legend()

        plt.show()

    def update_feature_list(self, phi_s_u, s_, s, ex):
        self.phi_s_u_list.append(phi_s_u)
        self.b_s_list.append(s - s_ - ex)


    def get_trajectory_3(self, x0, time_hor, s_u, s_w, param_u, mult_u, param_w):

        g_ = np.array([0, 0, g])  # gravity vector in inertial frame
        J = np.diag(np.array([self.I_xx, self.I_yy, self.I_zz]))  # inertia matrix
        drag = np.diag(np.array([self.Ax, self.Ay, self.Az]))  # drag force coefficients

        # ----------------------------------------- initial states -----------------------------------------------------
        x = np.array(x0)
        p = x[:3]  # position in inertial frame
        v = x[3:6]  # velocity in inertial frame
        q = x[6:10]  # quaternions
        omega = x[10:]  # angular velocity in body frame

        #  ------------------------------------- Storing the states - ---------------------------------------------
        self.px_list = [p[0]]
        self.py_list = [p[1]]
        self.pz_list = [p[2]]

        self.vx_list = [v[0]]
        self.vy_list = [v[1]]
        self.vz_list = [v[2]]

        self.q0_list = [q[0]]
        self.q1_list = [q[1]]
        self.q2_list = [q[2]]
        self.q3_list = [q[3]]

        self.wx_list = [omega[0]]
        self.wy_list = [omega[1]]
        self.wz_list = [omega[2]]

        # -----------------  random noise and disturbance generation ---------------------------------------------------
        U1_list, U2_list, U3_list, U4_list = generate_u(self.input, time_hor, s_u, mean=param_u[0], std=param_u[1], u_max=3.0, lb=param_u[0], ub=param_u[1])
        W1_list, W2_list, W3_list, W4_list, W5_list, W6_list = generate_w(self.distr, time_hor, s_w, mean=param_w[0], std=param_w[1], w_max=3.0, lb=param_w[0], ub=param_w[1])

        # ---------------------------------- PD controller gains  ------------------------------------------------------
        kp_z = 0.75
        kd_z = 1.25
        kp_phi = 0.03
        kp_theta = 0.03
        kp_psi = 0.03
        kd_phi = 0.00875
        kd_theta = 0.00875
        kd_psi = 0.00875

        # ---------------------------------   desired states   ---------------------------------------------------------
        pz_d = 1.
        vz_d = 0.
        q0_d, q1_d, q2_d, q3_d = euler_to_quaternion(0, 0, 0)

        for t in range(time_hor):

            s_ = np.array([v[0], v[1], v[2], omega[0], omega[1], omega[2]])

            q0 = q[0]
            q1 = q[1]
            q2 = q[2]
            q3 = q[3]

            omega1 = omega[0]
            omega2 = omega[1]
            omega3 = omega[2]

            # ------------------  noise in control input  (for exploration)  ----------------------------------------
            u1 = mult_u[0] * U1_list[t]
            u2 = mult_u[1] * U2_list[t]
            u3 = mult_u[2] * U3_list[t]
            u4 = mult_u[3] * U4_list[t]

            # ----------------   noise in control input  (for exploration)  -----------------------------------------
            w1 = W1_list[t]
            w2 = W2_list[t]
            w3 = W3_list[t]
            w4 = W4_list[t]
            w5 = W5_list[t]
            w6 = W6_list[t]

            # ----------------------------------------  PD control + noise  ------------------------------------------
            pi_z = kp_z * (pz_d - p[2]) + kd_z * (vz_d - v[2])
            f_c = np.array([0, 0, (5 + pi_z + u1)])

            qe1 = - q0_d * q1 - q3_d * q2 + q2_d * q3 + q1_d * q0
            qe2 = q3_d * q1 - q0_d * q2 - q1_d * q3 + q2_d * q0
            qe3 = - q2_d * q1 + q1_d * q2 - q0_d * q3 + q3_d * q0
            qe4 = q1_d * q1 + q2_d * q2 + q3_d * q3 + q0_d * q0

            pi_phi = -kd_phi * omega1 + kp_phi * qe1 * qe4
            pi_theta = -kd_theta * omega2 + 2 * kp_theta * qe2 * qe4
            pi_psi = -kd_psi * omega3 + 2 * kp_psi * qe3 * qe4
            tau_c = np.array([pi_phi + u2, pi_theta + u3, pi_psi + u4])

            # print('f', f_c)
            # print('tau', tau_c)
            # ------------------------------------------  Dynamic model ----------------------------------------------
            Q = np.array([[q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3),
                           2 * (q0 * q2 + q1 * q3)],
                          [2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
                           2 * (q2 * q3 - q0 * q1)],
                          [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3),
                           q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]])

            Omega = np.array([[0, -omega1, -omega2, -omega3],
                              [omega1, 0, omega3, -omega2],
                              [omega2, -omega3, 0, omega1],
                              [omega3, omega2, -omega1, 0]])

            # p_dot = v
            # v_dot = - g_ + (Q @ f_c - drag @ v) / self.m + np.array([w1, w2, w3])
            # q_dot = Omega @ q / 2
            # omega_dot = (np.linalg.inv(J) @ (tau_c + np.cross(omega, J @ omega)) + np.array([w4, w5, w6]))

            theta_star = np.array(
                [[1 / self.m, 0., 0., -self.Ax / self.m, 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 1 / self.m, 0., 0., -self.Ay / self.m, 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 1 / self.m, 0., 0., -self.Az / self.m, 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., (self.I_yy - self.I_zz) / self.I_xx, 0., 0., 1 / self.I_xx, 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., (self.I_zz - self.I_xx) / self.I_yy, 0., 0., 1 / self.I_yy, 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., (self.I_yy - self.I_xx) / self.I_zz, 0., 0., 1 / self.I_zz]])
            Qfc = Q @ f_c
            phi_s_u = np.array([Qfc[0], Qfc[1], Qfc[2],
                                v[0], v[1], v[2],
                                omega2 * omega3, omega1 * omega3, omega1 * omega2,
                                tau_c[0], tau_c[1], tau_c[2]])

            p_dot = v
            q_dot = Omega @ q / 2
            s_dot = - np.array([0., 0., g, 0., 0., 0.]) + theta_star @ phi_s_u + np.array([w1, w2, w3, w4, w5, w6])

            # -------------------------------------- Updating the states --------------------------------------------
            # p = p + dt * p_dot
            # v = v + dt * v_dot
            # q = (q + dt * q_dot) / (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
            # omega = omega + dt * omega_dot

            p = p + dt * p_dot
            q = (q + dt * q_dot) / (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
            s = s_ + dt * s_dot
            v = s[:3]
            omega = s[3:]

            self.update_feature_list(dt * phi_s_u, s_, s, - dt * np.array([0., 0., g, 0., 0., 0.]))

            # ------------------------------------- Storing the states ----------------------------------------------
            self.px_list.append(p[0])
            self.py_list.append(p[1])
            self.pz_list.append(p[2])

            self.vx_list.append(v[0])
            self.vy_list.append(v[1])
            self.vz_list.append(v[2])

            self.q0_list.append(q[0])
            self.q1_list.append(q[1])
            self.q2_list.append(q[2])
            self.q3_list.append(q[3])

            self.wx_list.append(omega[0])
            self.wy_list.append(omega[1])
            self.wz_list.append(omega[2])

