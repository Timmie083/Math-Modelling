import math
import simutils as su

import numpy as np

import plotter as pl

# Initial state from TLE (may be used later)
#ri, vi = state_from_tle_params(
#    e=0.0001406,
#    Revs_per_day=15.36545391,
#    Me=228.3364,
#    Omega=58.4052,
#    i=97.2852,
#    w=104.1529
#)

mu = 398600.4418 # Standard gravitational parameter [km**3/s**-2]
R_E = 6378.1363  # Radius of earth [km]
w_E = 7.292115e-5 # Angular speed of earth [rad/s]

###################################
# Assignment 2 | Helper functions #
###################################

# Conversion between anomalies
def mean_anomaly_from_eccentric_anomaly(E, e):
    """
    Converts eccentric anomaly into mean anomaly.
    :param E: Eccentric anomaly [radians]
    :param e: Eccentricity
    :return: Mean anomaly [radians]
    """
    return E - e*np.sin(E)

def mean_anomaly_from_true_anomaly(theta, e):
    """
    Converts true anomaly into mean anomaly.
    :param theta: True anomaly [radians]
    :param e: Eccentricity
    :return: Mean anomaly [radians]
    """
    return mean_anomaly_from_eccentric_anomaly(eccentric_anomaly_from_true_anomaly(theta, e), e)

def true_anomaly_from_eccentric_anomaly(E, e):
    """
    Converts eccentric anomaly into true anomaly.
    :param E: Eccentric anomaly [radians]
    :param e: Eccentricity
    :return: True anomaly [radians]
    """
    return 2 * math.atan(math.sqrt((1+e)/(1-e)) * math.tan(E/2))

def eccentric_anomaly_from_true_anomaly(theta, e):
    """
    Converts true anomaly into eccentric anomaly.
    :param theta: True anomaly [radians]
    :param e: Eccentricity
    :return: Eccentric anomaly [radians]
    """
    return 2 * math.atan(math.sqrt((1-e)/(1+e)) * math.tan(theta/2))

def eccentric_anomaly_from_mean_anomaly(Me, e, delta=1e-10, N=50):
    """
    Converts mean anomaly into eccentric anomaly using newtons method.
    :param Me: Mean anomaly [radians]
    :param e: Eccentricity
    :param delta: Tolerance for Newton iteration (default: 1e-10)
    :param N: Maximum number of iterations (default: 50)
    :return: Eccentric anomaly [radians]
    """
    # Initial guess
    if Me < np.pi:
        E = Me + e / 2
    else:
        E = Me - e / 2

    for _ in range(N):
        f = E - e*np.sin(E) - Me
        f_prime = 1 - e*np.cos(E)

        E_next = E - f / f_prime

        if abs(E_next - E) < delta:
            return E_next

        E = E_next

    raise RuntimeError("Did not converge")

# Orbital period
def orbital_period_from_semi_major_axis(a, u=mu):
    """
    Calculates orbital period using semi-major axis.
    :param a: Semi-major axis [km]
    :param u: Standard gravitational parameter (default: Earth's μ) [km**3/s**2]
    :return: Orbital period [s]
    """
    return 2 * math.pi * math.sqrt(a**3 / u)

def orbital_period_from_Revs_per_day(x):
    """
    Calculates orbital period from orbital revolution per day.
    :param x: Revolutions per day
    :return: Orbital period [s]
    """
    return 24 * 3600 / x

#TLE functions
def orbit_params_from_tle_params(tle):
    """
    Extracts orbital parameters from TLE text.
    :param tle: TLE text
    :return: List of orbital parameters:
             - epoch (YYDDD.DDDDDDDD)
             - inclination [degrees]
             - RAAN [degrees]
             - eccentricity
             - argument of perigee [degrees]
             - mean anomaly [degrees]
             - revolutions per day
    """
    line1, line2 = tle.splitlines()

    return {
        "epoch": line1[18:32].strip(),
        "inclination": float(line2[8:16]),
        "raan": float(line2[17:25]),
        "eccentricity": float("0." + line2[26:33].strip()),
        "arg_perigee": float(line2[34:42]),
        "mean_anomaly": float(line2[43:51]),
        "mean_motion": float(line2[52:63]),
    }

def tle_params_from_orbit_params(epoch, inclination, raan, eccentricity, arg_perigee, mean_anomaly, mean_motion):
    """
    Extracts TLE text from orbital parameters.
    :param epoch: Number of epochs
    :param inclination [degrees]
    :param RAAN: Right Ascension of the Ascending Node [degrees]
    :param eccentricity
    :param arg_perigee [degrees] 
    :param mean_anomaly [degrees]
    :param mean_motion 
    :return: TLE text:
    """
    # --- Line 1 ---
    line1 = (
        f"{epoch:>14}  "
    )

    # --- Line 2 ---
    inc = f"{inclination:8.4f}"
    raan = f"{raan:8.4f}"

    ecc = f"{eccentricity:.7f}".split('.')[1]

    argp = f"{arg_perigee:8.4f}"
    M = f"{mean_anomaly:8.4f}"
    n = f"{mean_motion:11.8f}"

    line2 = (
        f"{inc}"
        f"{raan}"
        f" {ecc}"
        f"{argp}"
        f"{M}"
        f"{n}"
    )

    return line1, line2

# Matrix functions
def rotation_matrix_from_classical_euler_sequence(Omega, i, w):
    """
    Returns the 3x3 rotation matrix for a classical Euler sequence.

    Rotations are applied in the following order:
    - Omega [RAAN] (around z-axis)
    - i     [Inclination] (around x-axis)
    - w     [Argument of perihelion] (around z-axis)

    :param Omega: Right Ascension of the Ascending Node (RAAN) [radians]
    :param i: Inclination [radians]
    :param w: Argument of perihelion [radians]
    :return: 3x3 rotation matrix
    """
    cos_O = np.cos(Omega)
    sin_O = np.sin(Omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_w = np.cos(w)
    sin_w = np.sin(w)

    R = np.array([
        [
            cos_O * cos_w - sin_O * sin_w * cos_i,
            -cos_O * sin_w - sin_O * cos_w * cos_i,
            sin_O * sin_i
        ],
        [
            sin_O * cos_w + cos_O * sin_w * cos_i,
            -sin_O * sin_w + cos_O * cos_w * cos_i,
            -cos_O * sin_i
        ],
        [
            sin_w * sin_i,
            cos_w * sin_i,
            cos_i
        ]
    ])

    return R

def rotation_matrix_from_roll_pitch_yaw_sequence(roll, pitch, yaw):
    """
    Returns the 3x3 rotation matrix for a roll-pitch-yaw (RPY) sequence.

    Rotations are applied in the following order:
    - Roll  (rotation about x-axis)
    - Pitch (rotation about y-axis)
    - Yaw   (rotation about z-axis)

    :param roll: Roll angle [radians]
    :param pitch: Pitch angle [radians]
    :param yaw: Yaw angle [radians]
    :return: 3x3 rotation matrix
    """
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])

    return R

# Quaternion functions
def quaternion_from_classical_euler_sequence(Omega, i, w):
    """
    Return the quaternion for a classical Euler sequence.

    Rotations are applied in the following order:
    - Omega [RAAN] (around z-axis)
    - i     [Inclination] (around x-axis)
    - w     [Argument of perihelion] (around z-axis)

    :param omega: Right Ascension of the Ascending Node (RAAN) [radians]
    :param i: Inclination [radians]
    :param w: Argument of perihelion [radians]
    :return: Quaternion representing the rotation
    """
    def quat_z(angle):
        return np.array([
            np.cos(angle / 2),
            0,
            0,
            np.sin(angle / 2)
        ])

    def quat_x(angle):
        return np.array([
            np.cos(angle / 2),
            np.sin(angle / 2),
            0,
            0
        ])

    def quat_mult(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    q_Omega = quat_z(Omega)
    q_i     = quat_x(i)
    q_w = quat_z(w)

    q = quat_mult(q_Omega, quat_mult(q_i, q_w))

    return q

def quaternion_from_roll_pitch_yaw_sequence(roll, pitch, yaw):
    """
    Return the quaternion for a roll-pitch-yaw (RPY) sequence.

    Rotations are applied in the following order:
    - Roll  (rotation about x-axis)
    - Pitch (rotation about y-axis)
    - Yaw   (rotation about z-axis)

    :param roll: Roll angle [radians]
    :param pitch: Pitch angle [radians]
    :param yaw: Yaw angle [radians]
    :return: Quaternion representing the rotation
    """
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy

    return np.array([w, x, y, z])

# Degree & radian functions
def rad2deg(rad):
    """
    Converts the given angle from radians to degrees.
    :param rad: Angle [radians]
    :return: Angle [degrees]
    """
    return rad * 180 / math.pi

def angle_wrap_degrees(deg):
    """
    Wraps the given angle in degrees to the range [0, 360].
    :param deg: Angle [degrees]
    :return: Equivalent angle in range [0, 360]
    """
    return deg % 360

def angle_wrap_radians(rad: float) -> float:
    """
    Wraps the given angle in radians to the range [0, 2pi].
    :param rad: Angle [radians]
    :return: Equivalent angle in range [0, 2pi]
    """
    return rad % (2 * math.pi)

# Polar coordinate functions
def polar2xyz(r: float, theta: float, out: np.ndarray=None) -> np.ndarray:
    """
    Converts polar coordinates in the orbital plane to 3D cartesian coordinates.

    The output lies in the XY-plane with z = 0.

    :param r: Radius (any consistent unit, e.g., km)
    :param theta: Angle [radians]
    :param out: Optional output array to store the result
    :return: 3-element NumPy array [x, y, z] with z = 0
    """
    if out is None:
        out = np.empty(3)

    out[0] = r * math.cos(theta)
    out[1] = r * math.sin(theta)
    out[2] = 0
    return out

###################################
# Assignment 2 | Algorithms       #
###################################

# Algorithm 1
def sidereal_angle(JD):
    """
    Calculates the Greenwich sidereal angle of the Earth from a Julian date.

    :param JD: Julian Date (days since J2000.0, can include fractional day)
    :return: Greenwich sidereal angle [radians]
    """
    # Julian centuries since J2000
    T = (JD - 2451545.0) / 36525.0

    # GMST in degrees
    theta_deg = (
        280.46061837
        + 360.98564736629 * (JD - 2451545.0)
        + 0.000387933 * T**2
        - (T**3) / 38710000.0
    )

    # Wrap to [0, 360]
    theta_deg = theta_deg % 360.0

    # Convert to radians
    theta_rad = np.radians(theta_deg)

    return theta_rad

# Algortihm 2
def state_from_orbit_params(h,e,theta,Omega,i,w, u=mu):
    """
    Computes the satellite's position and velocity vectors in Earth-Centered Inertial (ECI) frame
    from classical orbital elements.

    :param h: Specific angular momentum [km**2/s]
    :param e: Eccentricity
    :param theta: True anomaly [radians]
    :param omega: Right Ascension of Ascending Node (RAAN) [radians]
    :param i: Inclination [radians]
    :param w: Argument of perihelion [radians]
    :param u: Standard gravitational parameter (default: Earth's μ) [km**3/s**2]
    :return: Tuple of two np.ndarray elements:
             - Position vector in ECI frame [km]
             - Velocity vector in ECI frame [km/s]
    """
    # Position in perifocal frame
    r_p = (h**2 / u) / (1 + e * np.cos(theta)) * np.array([
        np.cos(theta),
        np.sin(theta),
        0
    ])

    # Velocity in perifocal frame
    v_p = (u / h) * np.array([
        -np.sin(theta),
        e + np.cos(theta),
        0
    ])

    # Rotation matrix
    R = rotation_matrix_from_classical_euler_sequence(Omega, i, w)

    r = R @ r_p
    v = R @ v_p

    return r, v

# Algorithm 3
def state_from_tle_params(e, Revs_per_day, Me, Omega, i, w):
    import numpy as np

    # Convert degrees → radians
    Me = np.radians(Me)
    Omega = np.radians(Omega)
    i = np.radians(i)
    w = np.radians(w)

    # Mean motion (rad/s)
    n = Revs_per_day * 2 * np.pi / (24 * 3600)

    # Gravitational parameter (Earth)
    mu = 398600.4418  # km^3/s^2

    # Semi-major axis
    a = (mu / n**2)**(1/3)

    # Angular momentum
    h = np.sqrt(mu * a * (1 - e**2))

    # Solve Kepler
    E = eccentric_anomaly_from_mean_anomaly(Me, e)

    # True anomaly
    theta = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )

    return state_from_orbit_params(h, e, theta, Omega, i, w)

# Algorithm 4
def orbit_params_from_state(ri,vi):
    """
    Calculates classical orbital elements from position and velocity vectors.
    :param ri: Position vector in ECI frame [km]
    :param vi: Velocity vector in ECI frame [km/s]
    :param u: Standard gravitational parameter (default: Earth's μ) [km**3/s**2]
    :return: Tuple containing orbital parameters:
             - Specific angular momentum [km**2/s]
             - Eccentricity
             - True anomaly [radians]
             - Right Ascension of Ascending Node (RAAN) [radians]
             - Inclination [radians]
             - Argument of perihelion [radians]
    """
    ri = np.array(ri)
    vi = np.array(vi)

    r_norm = np.linalg.norm(ri)
    v_norm = np.linalg.norm(vi)

    # Angular momentum
    h_vec = np.cross(ri, vi)
    h = np.linalg.norm(h_vec)

    # Inclination
    i = np.arccos(h_vec[2] / h)

    # Node vector
    k = np.array([0, 0, 1])
    n_vec = np.cross(k, h_vec)
    n = np.linalg.norm(n_vec)

    # Eccentricity vector
    e_vec = (1/mu) * ((v_norm**2 - mu/r_norm)*ri - np.dot(ri, vi)*vi)
    e = np.linalg.norm(e_vec)

    # RAAN
    Omega = np.arctan2(n_vec[1], n_vec[0])

    # Argument of perigee
    w = np.arctan2(
        np.dot(np.cross(n_vec, e_vec), h_vec) / h,
        np.dot(n_vec, e_vec)
    )

    # True anomaly
    theta = np.arctan2(
        np.dot(np.cross(e_vec, ri), h_vec) / h,
        np.dot(e_vec, ri)
    )

    return h, e, theta, Omega, i, w

#Algorithm 5
def orbit_propagation(ri, vi):
    h, e, theta, omega, i, w = orbit_params_from_state(ri, vi)

    # Get mean anomaly
    Me = mean_anomaly_from_eccentric_anomaly(eccentric_anomaly_from_true_anomaly(theta, e), e)

    # Mean motion
    a = h ** 2 / (mu * (1 - e ** 2))
    T = orbital_period_from_semi_major_axis(a)
    n = 2 * math.pi / T

    # Test
    pos_plot = np.concatenate(([0], ri))  # Initialize the plot data

    # Propagation loop
    dt = 1
    for t in range(0, int(T), dt):
        Me = angle_wrap_radians(Me + n * dt)
        E = eccentric_anomaly_from_mean_anomaly(Me, e)
        theta = true_anomaly_from_eccentric_anomaly(E, e)

        # Get the new ri, vi
        ri, vi = state_from_orbit_params(h, e, theta, omega, i, w)
        pos_plot = np.vstack((pos_plot, np.concatenate(([t], ri))))

    file = su.log_pos("assignment2_position", pos_plot)
    pos_plot = None  # Clear the data after its saved
    pl.line_plot(file)

# Algorithm 6
def epoch_to_julian_date(epoch):
    """
    Convert TLE epoch (YYDDD.DDDDDDDD) → Julian Date
    """

    epoch = float(epoch)

    year = int(epoch // 1000)
    day = epoch % 1000

    # Fix century
    year += 2000 if year < 57 else 1900

    # Julian Date at Jan 1 of that year
    JD_year_start = 367*year - int(7*(year + int((1+9)/12))/4) \
                    + int(275*1/9) + 1721013.5

    # Add day of year
    JD = JD_year_start + day - 1

    return JD

# Function for earth rotation
def earth_rotation_angle(theta0, t):
    """
    Earth rotation over time

    theta0 : initial sidereal angle
    t      : time since epoch (seconds)
    """
    w_E = 7.2921159e-5  # rad/s

    theta = theta0 + w_E * t

    return theta % (2 * np.pi)

###################################
# Assignment 3 | Algorithms       #
###################################

def get_orbit_energy_state(x, m, u=mu):
    """
    Calculates the total mechanical energy of a satellite in orbit from a full state vector.
    :param x: State vector containing position [km] and velocity [km/s]
    :param m: Mass of the satellite [kg]
    :param u: Standard gravitational parameter (default: Earth's μ) [km**3/s**2]
    :return: Total orbital energy [MJ]
    """
    return get_orbit_energy(x[:3], x[3:], m, u)

def get_orbit_energy(ri, vi, m, mu):
    """
    Calculates the total mechanical energy of a satellite in orbit from position and velocity vectors.
    :param ri: Position vector in ECI frame [km]
    :param vi: Velocity vector in ECI frame [km/s]
    :param m: Mass of the satellite [kg]
    :param u: Standard gravitational parameter (default: Earth's μ) [km**3/s**2]
    :return: Total orbital energy [MJ]
    """
    r = np.linalg.norm(ri).astype(float)
    v = np.linalg.norm(vi).astype(float)

    return m * (0.5*v**2 - mu / r)

def get_orbit_eccentricity_vector_state(x, u=mu):
    """
    Compute the orbital eccentricity vector from a full state vector.
    :param x: State vector containing position [km] and velocity [km/s]
    :param u: Standard gravitational parameter (default: Earth's μ) [km**3/s**2]
    :return: Eccentricity vector
    """
    return get_orbit_eccentricity_vector(x[:3], x[3:], u)

def get_orbit_eccentricity_vector(ri, vi, u=mu):
    """
    Compute the orbital eccentricity vector from position and velocity vectors.
    :param ri: Position vector in ECI frame [km]
    :param vi: Velocity vector in ECI frame [km/s]
    :param u: Standard gravitational parameter (default: Earth's μ) [km**3/s**2]
    :return: Eccentricity vector
    """
    r = np.linalg.norm(ri)
    hi = np.cross(ri, vi)

    return np.cross(vi, hi) / u - ri / r

def get_orbit_apoapsis(x, e, u=mu):
    """
    Compute the apoapsis distance of an orbit from a full state vector.
    :param x: State vector containing position [km] and velocity [km/s]
    :param e: Eccentricity (optional)
    :param u: Standard gravitational parameter (default: Earth's μ) [km**3/s**2]
    :return: Apoapsis distance [km]
    """
    ri = x[:3]
    vi = x[3:]

    # Calculate the orbital eccentricity vector if it is not given
    e = np.linalg.norm(get_orbit_eccentricity_vector(ri, vi, u)) if e is None else e

    return np.linalg.norm(np.cross(ri, vi)) ** 2 / (u * (1 - e))

def get_orbit_periapsis(x, e, u=mu):
    """
    Compute the periapsis distance of an orbit from a full state vector.
    :param x: State vector containing position [km] and velocity [km/s]
    :param e: Eccentricity (optional)
    :param u: Standard gravitational parameter (default: Earth's μ) [km**3/s**2]
    :return: Periapsis distance [km]
    """
    ri = x[:3]
    vi = x[3:]

    # Calculate the orbital eccentricity vector if it is not given
    e = np.linalg.norm(get_orbit_eccentricity_vector(ri, vi, u)) if e is None else e

    return np.linalg.norm(np.cross(ri, vi)) ** 2 / (u * (1 + e))



