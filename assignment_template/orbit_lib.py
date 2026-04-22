import numpy as np

R_E = 6378.1363
my = 398600.4418
r = R_E + 400
omega_E = 7.292115e-5

def mean_anomaly_from_eccentric_anomaly(E,e):
    Me = E - e*np.sin(E)
    return Me

def orbital_period_from_semi_major_axis(a):
    T = ((2*np.pi)/(np.sqrt(my)))*a**(3/2)
    return T

def orbital_period_from_Revs_per_day(Revs_per_day):
    T = (24*3600)/Revs_per_day
    return T

def eccentric_anomaly_from_true_anomaly(theta,e):
    E = 2 * np.atan(np.sqrt((1-e)/(1+e)) * np.tan(theta/2))
    return E

def true_anomaly_from_eccentric_anomaly(e):
    # Normalize M to [0, 2π]
    M = M % (2 * np.pi)

    if M < np.pi:
        E = M + e / 2
    else:
        E = M - e / 2

    theta = 2*np.atan*((np.sqrt((1+e)/(1-e))*np.tan(E/2)))

    return theta

def orbit_params_from_tle_params(tle):
    
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

def tle_params_from_orbit_params(epoch,inclination,raan,eccentricity,arg_perigee,mean_anomaly,mean_motion,):
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

def rotation_matrix_from_classical_euler_sequence(Omega,i,omega):

    cos_O = np.cos(Omega)
    sin_O = np.sin(Omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_w = np.cos(omega)
    sin_w = np.sin(omega)

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

def quaternion_from_classical_euler_sequence(Omega,i,omega):

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
    q_omega = quat_z(omega)

    q = quat_mult(q_Omega, quat_mult(q_i, q_omega))

    return q

def rotation_matrix_from_roll_pitch_yaw_sequence(yaw,pitch,roll):

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])

    return R

def quaternion_from_roll_pitch_yaw_sequence(yaw,pitch,roll):

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

def angle_wrap_radians(angle_in_radians):

    return angle_in_radians % (2 * np.pi)

def angle_wrap_degrees(angle_in_degrees):

    return angle_in_degrees % 360.0

def sidereal_angle(JD):

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

def state_from_orbit_params(h,e,theta,Omega,i,omega):

    # Position in perifocal frame
    r_p = (h**2 / my) / (1 + e * np.cos(theta)) * np.array([
        np.cos(theta),
        np.sin(theta),
        0
    ])

    # Velocity in perifocal frame
    v_p = (my / h) * np.array([
        -np.sin(theta),
        e + np.cos(theta),
        0
    ])

    # Rotation matrix
    R = rotation_matrix_from_classical_euler_sequence(Omega, i, omega)

    r = R @ r_p
    v = R @ v_p

    return r, v

def state_from_tle_params(e, Revs_per_day, Me, Omega, i, omega):
    import numpy as np

    # Convert degrees → radians
    Me = np.radians(Me)
    Omega = np.radians(Omega)
    i = np.radians(i)
    omega = np.radians(omega)

    # Mean motion (rad/s)
    n = Revs_per_day * 2 * np.pi / (24 * 3600)

    # Gravitational parameter (Earth)
    my = 398600.4418  # km^3/s^2

    # Semi-major axis
    a = (my / n**2)**(1/3)

    # Angular momentum
    h = np.sqrt(my * a * (1 - e**2))

    # Solve Kepler
    E = eccentric_anomaly_from_mean_anomaly(Me, e)

    # True anomaly
    theta = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )

    return state_from_orbit_params(h, e, theta, Omega, i, omega)

def orbit_params_from_state(ri,vi):

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
    e_vec = (1/my) * ((v_norm**2 - my/r_norm)*ri - np.dot(ri, vi)*vi)
    e = np.linalg.norm(e_vec)

    # RAAN
    Omega = np.arctan2(n_vec[1], n_vec[0])

    # Argument of perigee
    omega = np.arctan2(
        np.dot(np.cross(n_vec, e_vec), h_vec) / h,
        np.dot(n_vec, e_vec)
    )

    # True anomaly
    theta = np.arctan2(
        np.dot(np.cross(e_vec, ri), h_vec) / h,
        np.dot(e_vec, ri)
    )

    return h, e, theta, Omega, i, omega

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

def eccentric_anomaly_from_mean_anomaly(Me, e, tol=1e-10, max_iter=100):

    # Initial guess
    if Me < np.pi:
        E = Me + e / 2
    else:
        E = Me - e / 2

    for _ in range(max_iter):
        f = E - e*np.sin(E) - Me
        f_prime = 1 - e*np.cos(E)

        E_next = E - f / f_prime

        if abs(E_next - E) < tol:
            return E_next

        E = E_next

    raise RuntimeError("Did not converge")

#---ALGORITHM 5-------------------

def propagate_orbit_algorithm5(ri, vi, dt, steps):
    """
    Orbit propagation using Algorithm 5 (Keplerian propagation)

    Inputs:
        ri, vi : initial position & velocity (ECI)
        dt     : timestep (seconds)
        steps  : number of steps

    Returns:
        r_hist, v_hist : arrays of propagated states
    """

    # -------------------------
    # Step 1: Classical elements
    # -------------------------
    h, e, theta, Omega, i, omega = orbit_params_from_state(ri, vi)

    # -------------------------
    # Step 2: Alternative params
    # -------------------------
    # Semi-major axis
    a = h**2 / (my * (1 - e**2))

    # Mean motion
    n = np.sqrt(my / a**3)

    # Initial eccentric anomaly
    E = 2 * np.arctan2(
        np.sqrt(1 - e) * np.sin(theta / 2),
        np.sqrt(1 + e) * np.cos(theta / 2)
    )

    # Initial mean anomaly
    Me = E - e * np.sin(E)

    # Storage
    r_hist = []
    v_hist = []

    # -------------------------
    # Step 3: Propagation loop
    # -------------------------
    for _ in range(steps):

        # 1. Update mean anomaly
        Me = Me + n * dt
        Me = Me % (2 * np.pi)

        # 2. Solve Kepler 
        E = eccentric_anomaly_from_mean_anomaly(Me, e)

        # True anomaly
        theta = 2 * np.arctan2(
            np.sqrt(1 + e) * np.sin(E / 2),
            np.sqrt(1 - e) * np.cos(E / 2)
        )

        # 3. Compute state vectors
        r, v = state_from_orbit_params(h, e, theta, Omega, i, omega)

        r_hist.append(r)
        v_hist.append(v)

    return np.array(r_hist), np.array(v_hist)

# Initial state from TLE
ri, vi = state_from_tle_params(
    e=0.0001406,
    Revs_per_day=15.36545391,
    Me=228.3364,
    Omega=58.4052,
    i=97.2852,
    omega=104.1529
)

def earth_rotation_angle(theta0, t):
    """
    Earth rotation over time

    theta0 : initial sidereal angle
    t      : time since epoch (seconds)
    """
    omega_E = 7.2921159e-5  # rad/s

    theta = theta0 + omega_E * t

    return theta % (2 * np.pi)

# Orbital period
T = 24 * 3600 / 15.36545391

# Run propagation
dt = 10  # seconds
steps = int(T / dt)

epoch = 26097.16668981

JD0 = epoch_to_julian_date(epoch)

theta0 = sidereal_angle(JD0)

r_hist, v_hist = propagate_orbit_algorithm5(ri, vi, dt, steps)

def get_orbit_eccentricity_vector_state(x, my):
    return get_orbit_eccentricity_vector(x[:3], x[3:], my)

def get_orbit_eccentricity_vector(ri, vi, my):

    r = np.linalg.norm(ri)
    hi = np.cross(ri, vi)

    return np.cross(vi, hi) / my - ri / r

def get_orbit_apoapsis(x, e, my):

    ri = x[:3]
    vi = x[3:]

    # Calculate the orbital eccentricity vector if it is not given
    e = np.linalg.norm(get_orbit_eccentricity_vector(ri, vi, my)) if e is None else e

    return np.linalg.norm(np.cross(ri, vi)) ** 2 / (my * (1 - e))

def get_orbit_periapsis(x, e, my):

    ri = x[:3]
    vi = x[3:]

    # Calculate the orbital eccentricity vector if it is not given
    e = np.linalg.norm(get_orbit_eccentricity_vector(ri, vi, my)) if e is None else e

    return np.linalg.norm(np.cross(ri, vi)) ** 2 / (my * (1 + e))
