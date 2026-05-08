import numpy as np
import simutils as su
import orbit_lib as ol
import simulator as sim
import math

###################################
# Assignment 5 | Classes          # Assig 4 below
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

###################################
# Assignment 4 | Classes          #
###################################

class RigidBody(sim.BaseScenario):
    def __init__(self, r, v, m, q, w, J):
        
        self.r = r
        self.v = v
        self.m = m
        self.q = q
        self.omega = w
        self.J = J

        self.tau = np.zeros(3)
        self.F = np.zeros(3)

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

    def __init__(self, q_ib, w_b_ib, J, r=np.zeros(3), v=np.zeros(3), m=1, orbit=None, substeps=50):
        
        """ Assignment 4 __init__
        self.rb = RigidBody()

        self.k1 = 2
        self.k2 = 1
        """

        # Assignement 5.3 modification
        
        self.orbit = orbit

        if self.orbit is not None:
            r, v = self.orbit.get_state()

        # Rigid body
        self.body = RigidBody(r, v, m, q_ib, w_b_ib, J)

        # Substepping
        self.N = substeps + 1

        # ADCS
        self.ADCS = ADCS_PD(1e-5, 2e-4, 0, J)

    """ Assignment 4 init
    def init(self, t):
        self.rb.init(t)

        # Override initial conditions
        self.rb.q = np.array([1, 0, 0, 0])
        self.rb.omega = np.array([0, 0, 0])

        self.rb.J = np.diag([0.5, 0.5, 0.5])

        self.q_d = np.array([0.5, 0.5, 0.5, 0.5])
        self.omega_d = np.array([0.2, -0.1, 0.05])
    """

    def update(self, t_k, t_step):

        """ Assignment 4 version control
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
        """

        # Assignement 5.3 Modification

        if self.orbit:
            self.update_with_orbit(t_k, t_step)
        else:
            self.update_with_dynamics(t_k, t_step)


    # With orbit

    def update_with_orbit(self, t_k, t_step):

        dt = t_step / self.N

        for _ in range(self.N):

            # Propagate orbit
            self.orbit.propagate(dt)

            # Get orbit frame
            q_io, w_i_io, _ = self.orbit.get_orbit_frame()

            # Get body state
            q_ib = self.body.q
            w_b_ib = self.body.omega

            # ADCS
            self.ADCS.update(q_ib, w_b_ib, q_io, w_i_io)

            self.body.tau = self.ADCS.get_control()

            # Update rigid body
            self.body.update(t_k, dt)

    # No orbit

    def update_with_dynamics(self, t_k, t_step):

        dt = t_step / self.N

        for _ in range(self.N):

            # No reference frame, therefore zero desired
            q_io = np.array([1, 0, 0, 0])
            w_i_io = np.zeros(3)

            q_ib = self.body.q
            w_b_ib = self.body.omega

            self.ADCS.update(q_ib, w_b_ib, q_io, w_i_io)

            self.body.tau = self.ADCS.get_control()

            self.body.update(t_k, dt)

    def get_state(self):
        return self.body.r, self.body.v, self.body.q, self.body.omega

    def get_orbit_frame(self):
        if self.orbit:
            return self.orbit.get_orbit_frame()
        else:
            r, v, _, _ = self.body.get_state()
            return ol.orbit_frame_from_state(r, v)
        
    def get(self):
        return [
            ['satellite', self.body.r, self.body.q],
        ]
    

def main():

    sim_config = {
        't_0': 0,
        't_e': 5731,
        't_step': 2,
        'speed_factor': 1,
        'anim_dt': 0.04,
        'scale_factor': 1000,
        'visualise': True
    }

    J_matrix = np.array([
        [0.00146519, 0.00001703, -0.00000633],
        [0.00001703, 0.00151512, -0.00001598],
        [-0.00000633, -0.00001598, 0.00146333]
    ])

    orbit = ol.orbit_tle(
        n=15.07529327671380 * 2*np.pi / (24*3600),
        e=0.0034704,
        M_e=np.deg2rad(323.2390),
        O=np.deg2rad(81.8923),
        i=np.deg2rad(97.7929),
        w=np.deg2rad(37.1228)
    )

    scenario = Satellite(
        q_ib=np.array([1, 0, 0, 0]),
        w_b_ib=np.zeros(3),
        J=J_matrix,
        orbit=orbit,
        substeps=50
    )

    sim.create_and_start_simulation(sim_config, scenario)

if __name__ == "__main__":
    main()
