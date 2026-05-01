import numpy as np
import simutils as su
import sat_lib as sl
import orbit_lib as ol
import simulator as sim
import plotter as pl
import math

###################################
# Assignment 4 | Classes          #
###################################

class RigidBody(sim.BaseScenario):
    def __init__(self):
        
        self.q = None       #Quaternion [4]
        self.omega = None   #Angular velocity [3]
        self.tau = None     #Torque [3]
        self.J = None       #Inertia matrix [3x3]

        self.r = None       #Position
        self.v = None       #Velocity
        self.F = None       #Force
        self.m = None       #Mass

        self.RK4_x = None

    def init(self, t):
        
        # Rotation
        self.q = np.array([1.0, 0, 0, 0])
        self.omega = np.array([0, 0, 5])
        self.tau = np.array([0, 0, 0])
        self.J = np.array([
            [2,     1,      0],
            [1,     10,     0.1],
            [0,     0.1,    2.5]
        ])

        # Translation
        self.r = np.array([7000, 0, 0])   # [km]
        self.v = np.array([0, 7.5, 0])      # [km/s]
        self.F = np.array([0, 0, 0])      # Force [N]
        self.m = 1000.0                         # [kg]

        # Full state vector
        self.RK4_x = np.hstack((self.q, self.omega, self.r, self.v))

    def update(self, t, dt):

        def f_wrapper(t, x, ae):
            return self.f(t, x)

        self.RK4_x = su.step_RK4(dt, t, self.RK4_x, f_wrapper, ae=None)

        # Unpack
        self.q = self.RK4_x[0:4]
        self.omega = self.RK4_x[4:7]
        self.r = self.RK4_x[7:10]
        self.v = self.RK4_x[10:13]

        # Normalize quaternion
        self.q = self.q / np.linalg.norm(self.q)

    def f(self, t, x):

        q = x[0:4]
        omega = x[4:7]
        r = x[7:10]
        v = x[10:13]

        # -----------------------
        # ROTATION
        # -----------------------

        # Quaternion kinematics
        wx, wy, wz = omega

        Omega = np.array([
            [0,   -wx, -wy, -wz],
            [wx,   0,   wz, -wy],
            [wy,  -wz,  0,   wx],
            [wz,   wy, -wx,  0]
        ])

        q_dot = 0.5 * Omega @ q

        # Rigid body dynamics

        omega_dot = np.linalg.solve(self.J, self.tau - np.cross(omega, self.J @ omega))

        # -----------------------
        # TRANSLATION (NEW)
        # -----------------------

        r_dot = v

        r_norm = np.linalg.norm(r)

        a_gravity = -ol.mu * r / r_norm**3

        # External force contribution
        a_force = self.F / self.m

        v_dot = a_gravity + a_force

        return np.hstack((q_dot, omega_dot, r_dot, v_dot))
    
    def get(self):
        quat = su.Quaternion(self.q)
        return [
            ['satellite', self.r, quat],
    ]

class Satellite(sim.BaseScenario):

    def __init__(self):
        self.rb = RigidBody()

        self.k1 = 2
        self.k2 = 1

    def init(self, t):
        self.rb.init(t)

        # Override initial conditions
        self.rb.q = np.array([1, 0, 0, 0])
        self.rb.omega = np.array([0, 0, 0])

        self.rb.J = np.diag([0.5, 0.5, 0.5])

        self.q_d = np.array([0.5, 0.5, 0.5, 0.5])
        self.omega_d = np.array([0.2, -0.1, 0.05])

    def update(self, t, dt):

        q = self.rb.q
        omega = self.rb.omega

        # Quaternion error
        q_e = su.quat_mult(su.quat_conj(self.q_d), q)
        if q_e[0] < 0:
            q_e = -q_e

        q_v = q_e[1:4]

        R_e = su.quaternion_to_dcm(q_e)
        omega_d_body = R_e @ self.omega_d

        omega_err = omega - omega_d_body

        tau = -self.k1 * q_v - self.k2 * omega_err

        self.rb.tau = tau
        self.rb.update(t, dt)

    def get(self):
        return [
            ['satellite', np.array([0, 0, 0]), self.rb.q],
        ]

def main():

    sim_config = {
        't_0': 0,
        't_e': 500,
        't_step': 0.01,
        'speed_factor': 1,
        'anim_dt': 0.04,
        'scale_factor': 1,
        'visualise': True
    }

    scenario = Satellite()
    sim.create_and_start_simulation(sim_config, scenario)

if __name__ == "__main__":
    main()

###################################
# Assignment 5 | Classes          #
###################################

class ADCS_PD:

    def __init__(self, k1, k2, f_c, J):

        self.k1 = k1
        self.k2 = k2
        self.J = J
        self.tau = np.zeros(3)

    def update(self, q_ib, w_b_ib, q_io, w_i_io):

        # Quaternion error (q_error = q_id^{-1} ⊗ q_ib)

        q_e = su.quat_mult(su.quat_conj(q_io), q_ib)

        if q_e[0] < 0:
            q_e = -q_e

        q_v = q_e[1:4]

        # Angular velocity error

            # Rotation matrix from error quaternion
        R_e = su.quaternion_to_dcm(q_e)

            # Convert desired angular velocity into body frame
        w_d_body = R_e @ w_i_io

        w_err = w_b_ib - w_d_body

        # PD control law

        self.tau = -self.k1 * q_v - self.k2 * w_err

    def get_control(self):
        return self.tau