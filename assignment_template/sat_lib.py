import numpy as np
import simutils as su
import orbit_lib as ol
import simulator as sim
import math
import datetime as dt
import plotter as pl

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
        """
        Compute spacecraft rotational and translational state derivatives.

        Parameters
        ----------
        t : float
            Simulation time [s].

        x : numpy.ndarray
            State vector:

            x = [q0, q1, q2, q3,
                wx, wy, wz,
                rx, ry, rz,
                vx, vy, vz]

            where:
            - q : attitude quaternion (scalar-first)
            - omega : angular velocity [rad/s]
            - r : position vector [km]
            - v : velocity vector [km/s]

        Returns
        -------
        x_dot : numpy.ndarray
            Time derivative of the state vector.
        """

        # State extraction
        q = x[0:4]
        omega = x[4:7]
        r = x[7:10]
        v = x[10:13]

        # -------------------------
        # Quaternion kinematics
        # -------------------------

        wx, wy, wz = omega

        Omega = np.array([
            [0,   -wx, -wy, -wz],
            [wx,   0,   wz, -wy],
            [wy,  -wz,  0,   wx],
            [wz,   wy, -wx,  0]
        ])

        q_dot = 0.5 * Omega @ q

        # -------------------------
        # Rotational dynamics
        # -------------------------

        omega_dot = np.linalg.solve(
            self.J,
            self.tau - np.cross(omega, self.J @ omega)
        )

        # -------------------------
        # Translational dynamics
        # -------------------------

        r_dot = v

        r_norm = np.linalg.norm(r)

        a_gravity = -ol.mu * r / r_norm**3

        a_force = self.F / self.m

        v_dot = a_gravity + a_force

        return np.hstack((q_dot, omega_dot, r_dot, v_dot))
    
    def get(self):
        quat = su.Quaternion(self.q)
        return [['satellite', self.r, quat],]

class Satellite(sim.BaseScenario):

    def __init__(self, q_ib, w_b_ib, J, r=np.zeros(3), v=np.zeros(3), m=1, orbit=None, substeps=50):

        self.groundtrack = []

        # Assignement 5.3 modification
        
        self.orbit = orbit

        if self.orbit is not None:
            r, v = self.orbit.get_state()
        else:
            r = np.zeros(3)
            v = np.zeros(3)

        # Rigid body
        self.body = RigidBody(r, v, m, q_ib, w_b_ib, J)

        # Substepping
        self.N = substeps

        # ADCS
        self.ADCS = ADCS_PD(1e-5, 2e-4, 0, J)

        tle = su.read_TLE_file("Assignment5_TLE.txt", satellite_name=None)

        print(tle)

        

        JD = ol.epoch_to_julian_date(epoch)

        self.sidereal = ol.sidereal_angle(JD)

        self.theta_E = 0 + self.sidereal

        self.q_E = su.Quaternion([1, 0, 0, 0])

    def update(self, t_k, t_step):

        self.theta_E = (t_k * ol.w_E) + self.sidereal

        temp = ol.polar2xyz(1, self.theta_E / 2)

        self.q_E = su.Quaternion([temp[0], 0, 0, temp[1]])

        # Assignement 5.3 Modification

        if self.orbit:
            self.update_with_orbit(t_k, t_step)
        else:
            self.update_with_dynamics(t_k, t_step)

        # ECI position
        r_eci = self.body.r

        # Convert to ECEF
        r_ecef = ol.eci_to_ecef(r_eci, self.theta_E)

        # Convert to geocentric coordinates
        _, lon, lat = ol.geocentric_from_xyz(r_ecef)

        # Store
        self.groundtrack.append([t_k, lon, lat])

    # With orbit

    def update_with_orbit(self, t_k, t_step):

        dt = t_step / self.N

        for _ in range(self.N):

            self.orbit.propagate(dt)

            # Update rigid-body translational state
            self.body.r, self.body.v = (self.orbit.get_state())

            r_ecef = ol.eci_to_ecef(self.body.r, self.theta_E)

            _, lon, lat = ol.geocentric_from_xyz(r_ecef)

            self.groundtrack.append([t_k, lon, lat])

            q_io, w_i_io, _ = (self.orbit.get_orbit_frame())

            q_ib = self.body.q

            w_b_ib = self.body.omega

            self.ADCS.update(q_ib, w_b_ib, q_io, w_i_io)

            self.body.tau = (self.ADCS.get_control())

            self.body.update(t_k, dt)

    # No orbit

    def update_with_dynamics(self, t_k, t_step):

        dt = t_step / self.N

        for _ in range(self.N):

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
            ['earth', np.zeros(3), self.q_E],
            ['ECEF frame', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()],
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

    tle_list = su.read_TLE_file(
        "Assignment5_TLE.txt"
    )

    tle = tle_list[0]

    orbit = ol.orbit_pkepler(tle)

    tnow = dt.datetime.utcnow()

    JD_now = ol.datetime_to_julian_date(
        tnow.year,
        tnow.month,
        tnow.day,
        tnow.hour,
        tnow.minute,
        tnow.second
    )

    JD_epoch = ol.tle_epoch_to_julian(
        tle["epoch"]
    )

    delta_t = (
        JD_now - JD_epoch
    ) * 86400

    orbit.propagate_time(delta_t)

    scenario = Satellite(
        q_ib=np.array([1, 0, 0, 0]),
        w_b_ib=np.zeros(3),
        J=J_matrix,
        orbit=orbit,
        substeps=50
    )

    sim.create_and_start_simulation(
        sim_config,
        scenario
    )

    np.savetxt(
        "groundtrack.txt",
        np.array(scenario.groundtrack)
    )

if __name__ == "__main__":
    main()

    data = np.loadtxt("groundtrack.txt")

    t, lon, lat = data.T

    pl.plot_ground_track(
        lon,
        lat,
        degrees=False
    )

###################################
# Assignment 7 | Classes          #
###################################

# Sensor classes

class gyro:

    def __init__(self, q_bs=su.Quaternion(), p_b=np.array([0, 0, 0]), z0=np.zeros(3), mu=0.0, Q=0.1, params=None):

        self.q_bs = q_bs            # sensor orientation wrt body
        self.p = p_b                # sensor position
        self.mu = mu                # mean noise
        self.Q = Q                  # variance
        self.z = z0                 # current measurement

        self.b_g = np.zeros(3)      # gyro bias
        
        self.Q_bias = 0             # optional bias random walk variance

        if params is not None:
            if "b_g" in params:
                self.b_g = params["b_g"]

            if "Q_bias" in params:
                self.Q_bias = params["Q_bias"]


    def update(self, t, t_step, q_ib, w_b_ib, r_i, v_i):

        # bias random walk
        bias_noise = np.random.normal(loc=0.0, scale=np.sqrt(self.Q_bias), size=3)

        self.b_g += bias_noise * t_step

        # rotate body angular velocity into sensor frame
        w_s = self.q_bs.conjugated().rotate(w_b_ib)

        # measurement noise
        noise = np.random.normal(loc=self.mu, scale=np.sqrt(self.Q), size=3)

        # gyro output
        self.z = w_s + self.b_g + noise

    def output(self, body_frame=False):

        if body_frame:
            # rotate measurement back to body frame
            return self.q_bs.rotate(self.z)

        return self.z
    
class magnetometer:

    def __init__(self, q_bs=su.Quaternion(), p_b=np.array([0, 0, 0]), z0=np.zeros(3), mu=0, Q=0.4, params=None):

        self.q_bs = q_bs
        self.p = p_b

        self.mu = mu
        self.Q = Q

        self.z = z0

        # hard iron bias
        self.b_B = np.zeros(3)

        # soft iron matrix
        self.M_B = np.eye(3)

        if params is not None:

            if "b_B" in params:
                self.b_B = params["b_B"]

            if "M_B" in params:
                self.M_B = params["M_B"]

        tle = su.read_TLE_file("Assignment5_TLE.txt", satellite_name=None)
        epoch = tle[0]["epoch"]
        self.JD0 = ol.epoch_to_julian_date(epoch)

    def update(self, t, t_step, q_ib, w_b_ib, r_i, v_i):

        # Convert simulation time to Julian Date
        # Replace JD0 with your simulation start Julian date
        JD = self.JD0 + t / 86400.0

        # Earth magnetic field in inertial frame
        B_i = ol.magnetic_field_dipole(r_i, JD)

        # Quaternion from inertial -> sensor frame
        q_is = q_ib @ self.q_bs

        # Rotate magnetic field into sensor frame
        B_s = q_is.conjugated().rotate(B_i)

        # Add gaussian noise
        noise = np.random.normal(loc=self.mu, scale=np.sqrt(self.Q), size=3)

        # Full measurement model
        self.z = self.M_B @ B_s + self.b_B + noise

    def output(self, body_frame=False):

        if body_frame:
            return self.q_bs.rotate(self.z)

        return self.z

class fine_sun_sensor:

    def __init__(self, q_bs=su.Quaternion(), p_b=np.array([0, 0, 0]), z0=np.zeros(3), mu=0, Q=0.2, params=None):

        self.q_bs = q_bs
        self.p = p_b

        self.mu = mu
        self.Q = Q

        self.z = z0

        # field of view
        self.alpha = np.pi

        if params is not None:

            if "alpha" in params:
                self.alpha = params["alpha"]

        tle = su.read_TLE_file("Assignment5_TLE.txt", satellite_name=None)
        epoch = tle[0]["epoch"]
        self.JD0 = ol.epoch_to_julian_date(epoch)


    def update(self, t, t_step, q_ib, w_b_ib, r_i, v_i):

        # simulation JD
        JD = self.JD0 + t / 86400.0

        # Sun vector in inertial frame
        s_i = ol.sun_vector(JD)

        # normalize
        s_i_hat = s_i / np.linalg.norm(s_i)

        # inertial -> sensor quaternion
        q_is = q_ib @ self.q_bs

        # rotate sun vector into sensor frame
        s_s = q_is.conjugated().rotate(s_i_hat)

        x, y, z = s_s

        # angle from sensor boresight (+z axis)
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)

        # check field of view
        if z > 0 and theta < self.alpha / 2:

            noise = np.random.normal(loc=self.mu, scale=np.sqrt(self.Q), size=3)

            self.z = s_s + noise

        else:

            # sun not visible
            self.z = np.zeros(3)

    def output(self, body_frame=False):

        if body_frame:
            return self.q_bs.rotate(self.z)

        return self.z

# Attitude determination classes

class TRIAD:

    def __init__(self, params=None):

        self.params = params


    def estimate_attitude(self, M_B, M_A):
        """
        M_B : list of vectors in body frame
        M_A : list of vectors in reference/orbital frame

        Returns:
            q_BA : quaternion rotating A -> B
        """
        # use first two vectors
        a_b = M_B[0]
        b_b = M_B[1]

        a_a = M_A[0]
        b_a = M_A[1]

        # normalize inputs
        a_b = a_b / np.linalg.norm(a_b)
        b_b = b_b / np.linalg.norm(b_b)

        a_a = a_a / np.linalg.norm(a_a)
        b_a = b_a / np.linalg.norm(b_a)

        # Construct TRIAD basis in frame A

        t1_a = a_a

        t2_a = np.cross(t1_a, b_a)
        t2_a = t2_a / np.linalg.norm(t2_a)

        t3_a = np.cross(t2_a, t1_a)

        # Construct TRIAD basis in frame B

        t1_b = a_b

        t2_b = np.cross(t1_b, b_b)
        t2_b = t2_b / np.linalg.norm(t2_b)

        t3_b = np.cross(t2_b, t1_b)

        # Build rotation matrices

        T_A = np.column_stack((t1_a, t2_a, t3_a))
        T_B = np.column_stack((t1_b, t2_b, t3_b))

        # Rotation from A -> B

        R_BA = T_B @ T_A.T

        # Convert to quaternion

        q_BA = su.dcm_to_quaternion(R_BA)

        return q_BA

class Davenport:

    def __init__(self, params=None):
        self.params = params

    def estimate_attitude(self, M_B, M_A, weights=None):
        """
        Davenport q-method attitude estimation.

        Args:
            M_B : list of vectors in body frame (u_b^i)
            M_A : list of vectors in reference frame (u_o^i)
            weights : list of weights a_i (optional). If None, uses uniform weights.

        Returns:
            q_bo : quaternion corresponding to optimal attitude estimate
        """

        N = len(M_B)

        M_B = [np.asarray(v, dtype=float) for v in M_B]
        M_A = [np.asarray(v, dtype=float) for v in M_A]

        # normalize input vectors (important for Wahba problem formulation)
        M_B = [v / np.linalg.norm(v) for v in M_B]
        M_A = [v / np.linalg.norm(v) for v in M_A]

        # set weights
        if weights is None:
            weights = np.ones(N) / N
        else:
            weights = np.asarray(weights, dtype=float)

        # initialize terms
        lam0 = np.sum(weights)
        B = np.zeros((3, 3))
        z = np.zeros(3)

        # compute B and z
        for i in range(N):
            u_b = M_B[i].reshape(3, 1)   # column vector
            u_o = M_A[i].reshape(1, 3)   # row vector

            B += weights[i] * (u_b @ u_o)
            z += weights[i] * np.cross(M_B[i], M_A[i])

        # Davenport K matrix
        trB = np.trace(B)

        K = np.zeros((4, 4))
        K[0, 0] = trB
        K[0, 1:] = z
        K[1:, 0] = z
        K[1:, 1:] = B + B.T - trB * np.eye(3)

        # eigen decomposition
        evals, evecs = np.linalg.eig(K)

        # dominant eigenvector (largest eigenvalue)
        max_idx = np.argmax(evals)
        q_bo = evecs[:, max_idx]

        # normalize quaternion
        q_bo = q_bo / np.linalg.norm(q_bo)

        return q_bo