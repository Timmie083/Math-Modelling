import numpy as np
import simutils as su
import sat_lib as sl
import orbit_lib as ol
import simulator as sim
import plotter as pl
import math

class RigidBody(sim.BaseScenario):
    def __init__(self):
        
        self.q = None       #Quaternion [4]
        self.omega = None   #Angular velocity [3]
        self.tau = None     #Torque [3]
        self.J = None       #Inertia matrix [3x3]
        self.RK4_x = None


    def init(self, t):

        self.q = np.array([1.0, 0, 0, 0])
        self.omega = np.array([0, 0, 5])
        self.tau = np.array([0, 0, 0])
        self.J = np.array([
            [2,     1,      0],
            [1,     10,     0.1],
            [0,     0.1,    2.5]
        ])

        self.RK4_x = np.hstack((self.q, self.omega))

    def update(self, t, dt):

        def f_wrapper(t, x, ae):
            return self.f(t, x)

        self.RK4_x = su.step_RK4(dt, t, self.RK4_x, f_wrapper, ae=None)

        self.q = self.RK4_x[0:4]
        self.omega = self.RK4_x[4:7]

        self.q = self.q / np.linalg.norm(self.q)

    def f(self, t, x):

        q = x[0:4]
        omega = x[4:7]

        # ----- Quaternion kinematics -----
        wx, wy, wz = omega

        Omega = np.array([
            [0,   -wx, -wy, -wz],
            [wx,   0,   wz, -wy],
            [wy,  -wz,  0,   wx],
            [wz,   wy, -wx,  0]
        ])

        q_dot = 0.5 * Omega @ q

        # ----- Rigid body dynamics -----
        J = self.J
        tau = self.tau

        omega_dot = np.linalg.solve(J, tau - np.cross(omega, J @ omega))

        return np.hstack((q_dot, omega_dot))
    
    def get(self):
        quat = su.Quaternion(self.q)
        return [
            ['satellite', np.array([0, 0, 0]), quat],
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