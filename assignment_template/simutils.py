import datetime as dt
import numpy as np
from vispy.scene import MatrixTransform as Mat4
from vispy.util.quaternion import Quaternion as Quat

import orbit_lib as ol
import assignment as assig

class Error(Exception):
    pass

class InvalidConstruction(Error):
    def __init__(self,message):
        self.message=message

class Quaternion:
    def __init__(self, arg1 = None, arg2 = None):
        if arg1 is None and arg2 is None:
            self.q = np.array([1,0,0,0])
        elif arg1 is not None and arg2 is None:
            if type(arg1) is Quaternion:
                self.q = np.array(arg1.q)
            elif len(arg1) == 4:
                self.q = np.array(arg1)
            elif len(arg1) == 3:
                self.q = np.array([0,*arg1])
            else:
                raise InvalidConstruction("Wrong initialization, expects one of:\narg1=None,arg2=None\narg1=Quaternion,arg2=None\narg1=list[4],arg2=None\narg1=list[3],arg2=None\narg1=float,arg2=list[3]\n")
        elif arg1 is not None and arg2 is not None:
            if len(arg2) == 3:
                mag = np.sqrt(arg2[0]**2.0+arg2[1]**2.0+arg2[2]**2.0)
                self.q = np.array([np.cos(arg1/2.0),*(np.sin(arg1/2.0)/mag*np.array(arg2))])
            else:
                raise InvalidConstruction("Wrong initialization, expects one of:\narg1=None,arg2=None\narg1=Quaternion,arg2=None\narg1=list[4],arg2=None\narg1=list[3],arg2=None\narg1=float,arg2=list[3]\n")
        else:
            raise InvalidConstruction("Wrong initialization, expects one of:\narg1=None,arg2=None\narg1=Quaternion,arg2=None\narg1=list[4],arg2=None\narg1=list[3],arg2=None\narg1=float,arg2=list[3]\n")
    
    def __len__(self):
        return len(self.q)
        
    def __repr__(self):
        return "Quaternion: [{}]".format(",".join([str(x) for x in self.q]))
    
    def __getitem__(self,index):
        if type(index) == slice:
            if index.stop < index.start:
                raise IndexError("starting index should be smaller than ending index")
            elif index.start in range(0,len(self)+1) and index.stop in range(0,len(self)+1):
                return np.array([self[i] for i in range(index.start,index.stop+1)])
            else:
                raise IndexError("Indexes out of bounds")
        else:
            if index > 3:
                raise IndexError("Index out of bounds")
            else:
                return self.q[index]

    def __add__(self,other):
        return Quaternion(self.q+other.q)
    
    def __sub__(self,other):
        return Quaternion(self.q-other.q)
    
    def __mul__(self,other):
        return Quaternion(self.q*other)
            
    def __rmul__(self,other):
        return self*other
    
    def __truediv__(self,other):
        return 1/other*self

    def __matmul__(self,other):
        return Quaternion([self[0]*other[0]-np.dot(self[1:3],other[1:3]), *(self[0]*other[1:3]+other[0]*self[1:3]+np.cross(self[1:3],other[1:3]))])
    
    def inverted(self):
        mag = self.magnitude()
        if mag < 1e-9:
            raise IndexError("Magnitude is zero")
        return 1.0/mag**2.0*self.conjugated()
    
    def conjugated(self):
        return Quaternion([self[0],*(-self[1:3])])

    def normalized(self):
        mag = self.magnitude()
        return Quaternion(self.q/mag)
    
    def invert(self):
        mag = self.magnitude()
        if mag < 1e-9:
            raise IndexError("Magnitude is zero")
        self.q /= mag**2.0
    
    def conjugate(self):
        self.q = np.array([self[0],*(-self[1:3])])
    
    def normalize(self):
        mag = self.magnitude()
        if mag < 1e-9:
            raise IndexError("Magnitude is zero")
        self.q = self.q/mag
        
    def magnitude(self):
        return np.linalg.norm(self.q)

    def rotate(self, u):
        v = self@Quaternion(u)@self.conjugated()
        return v[1:3]

def read_TLE_file(file_name,satellite_name=''):
  def validate_entry(Name,line1,line2):
    if not Name[0].isalpha():
      return False
    if not line1[0].startswith("1") or not len(line1) == 9:
      return False
    if not line2[0].startswith("2") or not len(line2) == 8:
      return False
    return True

  tle_data = []
  with open(file_name) as f:
    file_contents = f.readlines()
  if len(file_contents) < 3:
    print("Error reading file\nRequired format is:\nAAAAAAAAAAAAAAAAAAAAAAAA\n1 NNNNNU NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN\n2 NNNNN NNN.NNNN NNN.NNNN NNNNNNN NNN.NNNN NNN.NNNN NN.NNNNNNNNNNNNNN\nfor each entry")
    return tle_data

  for i in range(0,len(file_contents),3):
    if(satellite_name in file_contents[i]):
      Name = file_contents[i].strip()
      line1 = file_contents[i+1].strip().split()
      line2 = file_contents[i+2].strip().split()
      if validate_entry(Name,line1,line2):
        epoch = float(line1[3])
        e = float("0."+line2[4])
        rev = float(line2[7])
        Me = float(line2[6])
        i = float(line2[2])
        O = float(line2[3])
        w = float(line2[5])
        tle_data.append((Name,epoch,e,rev,Me,i,O,w))
      else:
        print("Error reading entry:\n{}{}{}".format(file_contents[i],file_contents[i+1],file_contents[i+2]))
        break
  return tle_data

def read_obj(fname):
    verts = []
    vcols = []
    faces = []
    with open(fname,'r') as f:
        for line in f:
            if line.startswith('v '):
                d = [float(x) for x in line.split(' ')[1:]]
                verts.append(d[0:3])
                if len(d) > 3:
                    vcols.append(d[3:])
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0])-1 for x in line.split(' ')[1:]])
            else:
                pass
    return np.array(verts),np.array(vcols),np.array(faces)

def rotscaleloc_to_vispy(pos=None,quat=None,Rot=None,Eul=None,scale=None):
	if quat is not None:
		q = Quat(w=quat[0],x=quat[1],y=quat[2],z=quat[3])
		H = Mat4(q.conjugate().get_matrix())
	elif Rot is not None:
		p = np.array([[0,0,0]]).T
		HT = np.vstack(((np.hstack((Rot,p)),np.array([[0,0,0,1]]))))
		H = Mat4(HT.T)
	elif Eul is not None:
		q = Quat.create_from_euler_angles(Eul[2],Eul[1],Eul[0])
		H = Mat4(q.conjugate().get_matrix())
	else:
		H = Mat4()
	if scale is not None:
		H.scale((scale,scale,scale))
	if pos is not None:
		H.translate(pos)
	return H

def H_to_Rp(H):
    return H.matrix[:3,:3].T,H.matrix[-1][:3]

def log_pos(name,pos,path='data/'):
    #file_name = path + name + '_' + dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.txt'
    file_name = path + name + '.txt'

    print("logged: "+file_name)

    open(file_name, 'a').close()
    np.savetxt(file_name,pos)

    return file_name

###################################
# Assignment 3 | Algorithms       #
###################################

# f(t, x)
def two_body(t, x, ae: np.ndarray=None, u=ol.mu): 
    """
    Compute the time derivative of the state vector for the classical two-body problem.

    The state vector x contains both the position and velocity vectors,
    formatted as: [rx, ry, rz, vx, vy, vz].

    :param t: Current time [s]
    :param x: State vector containing both position vector and velocity vector [km | km/s]
    :param ae: External acceleration vector (default: 0) [km/s**2]
    :param u: Standard gravitational parameter (default: Earth's μ) [km**3/s**2]
    :return: State vector with the time derivative of x [km/s | km/s**2]
    """
    ri = x[:3] 
    vi = x[3:] 

    r = np.linalg.norm(ri)
    ai = -u/r**3 * ri + (np.zeros(3) if ae is None else ae)

    return np.concatenate([vi, ai])

# Numeric solver functions
def step_euler(h,t_k,x_k,f):
    """
    Performs one step of the explicit Euler method for a first-order ODE.

    The state vector x contains both the position and velocity vectors,
    formatted as: [rx, ry, rz, vx, vy, vz].

    :param h: Time step size [s]
    :param t_k: Current time [s]
    :param x_k: State vector containing position [km] and velocity [km/s]
    :param f: Function f(t, x) returning dx/dt
    :return: State vector at time t + h [km | km/s]
    """
    return x_k + h * f(t_k,x_k)

def step_leapfrog(h,t_k,x_k,f):
    """
    Performs one step of the explicit Leapfrog method for a first-order ODE.

    The state vector x contains both the position and velocity vectors,
    formatted as: [rx, ry, rz, vx, vy, vz].

    :param h: Time step size [s]
    :param t_k: Current time [s]
    :param x_k: State vector containing position [km] and velocity [km/s]
    :param f: Function f(t, x) returning dx/dt
    :return: State vector at time t + h [km | km/s]
    """
    # Current values
    dx = f(t_k, x_k) # [vx, vy, vz, ax, ay, az]
    v_half = dx[:3] + 0.5 * dx[3:] * h

    # New values
    r_new  = x_k[:3] + v_half * h
    v_new = 2 * v_half - x_k[3:]

    return np.concatenate([r_new, v_new])

def step_verlet(h,t_k,x_k,x_prev,f):
    if x_prev is None: 
        r_k = x_k[:3]
        v_k = x_k[3:]

        a_k = f(t_k, x_k)[3:]
        r_next = r_k + v_k * h + 0.5 * a_k * h**2

        return np.concatenate([r_next, v_k])

    # Extract positions
    r_k = x_k[:3]
    r_prev = x_prev[:3]

    # Get acceleration from f
    a_k = f(t_k, x_k)[3:]

    # Verlet position update
    r_next = 2 * r_k - r_prev + a_k * h ** 2

    # AI Slop!!! No idea of its accuracy but velocity is optional in verlet
    v_next = (r_next - r_prev) / (2 * h)
    return np.concatenate([r_next, v_next])

def step_RK4(h,t_k,x_k,f,ae):

    t1 = t_k
    t2 = t1 + 0.5 * h
    t3 = t2
    t4 = t1 + h

    x1 = x_k
    x2 = x1 + 0.5 * h * f(t1,x1,ae)
    x3 = x1 + 0.5 * h * f(t2,x2,ae)
    x4 = x1 + h * f(t3,x3,ae)

    f1 = f(t1,x1,ae)
    f2 = f(t2,x2,ae)
    f3 = f(t3,x3,ae)
    f4 = f(t4,x4,ae)

    return x1 + (h/6) * (f1+2*f2+2*f3+f4)

###################################
# Assignment 4 | Algorithms       #
###################################

def quaternion_to_dcm(q):
    """
    Transformation between quaternion to Direction Cosine Matrix

    :param q: Quaternion vector [q0, q1, q2, q3]
    :return: DCM 3x3 matrix, R
    """
    q0, q1, q2, q3 = q

    return np.array([
        [q0**2 + q1**2 - q2**2 - q3**2,   2*(q1*q2 + q0*q3),           2*(q1*q3 - q0*q2)],
        [2*(q1*q2 - q0*q3),               q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 + q0*q1)],
        [2*(q1*q3 + q0*q2),               2*(q2*q3 - q0*q1),           q0**2 - q1**2 - q2**2 + q3**2]
    ])

def axis_angle_to_quaternion(theta, u):
    """
    Transformation between axis angle to quaternion

    :param theta: [Degrees]
    :param u: Unit-vector
    :return: Quaternion vector [q0, q1, q2, q3]
    """
    arg = np.array[u*np.sin(theta/2)]
    return np.array([
        [np.cos(theta/2)]
        [arg[0]]
        [arg[1]]
        [arg[2]]
    ])

def axis_angle_to_dcm(theta, u):
    """
    Transformation between axis angle to Direction Cosine Matrix

    :param theta: [Degrees]
    :param u: Unit-vector
    :return: DCM 3x3 matrix, R
    """
    S = np.array([
        [0,     -u[2],  u[1]]
        [u[2],  0,      -u[0]]
        [-u[1], u[0],   0]
    ])

    I = np.array([
        [1]
        [0]
        [0]
    ])

    return I + np.sin(theta)*S + (1 - np.cos(theta))*S**2

def dcm_to_quaternion(R):
    """
    Transformation between Direction Cosine Matrix to quaternion

    :param R: DCM 3x3 matrix
    :return: Quaternion vector [q0, q1, q2, q3]
    """
    q = np.zeros(4)
    trR = np.linalg.trace(R)
    if trR > 0:
        q[0] = 0.5 * np.sqrt(1+trR)
        q[1] = 1/(4 * q[0]) * (R[1,2]-R[2,1])
        q[2] = 1/(4 * q[0]) * (R[2,0]-R[0,2])
        q[3] = 1/(4 * q[0]) * (R[0,1]-R[1,0])
    else:
        D = R.diagonal()
        i, j, k = np.roll (np.arange(3), -np.argmax(D))
        q[i+1] = 0.5 * np.sqrt(1 + R[i,i] - R[j,j] - R[k,k])
        q[j+1] = 1/(4 * q[i+1]) * (R[i,j]-R[j,i])
        q[k+1] = 1/(4 * q[i+1]) * (R[i,k]-R[k,i])
        q[0] = 1/(4 * q[i+1]) * (R[j,k]-R[k,j])
    return Quaternion(np.sign(q[0]) * q).conjugated()

def euler_to_quaternion(roll, pitch, yaw):
    """
    Transformation between Euler angles (roll, pitch, yaw) to quaternion

    :param roll: Roll angle [degrees]
    :param pitch: Pitch angle [degrees]
    :param yaw: Yaw angle [degrees]
    :return: Quaternion vector [q0, q1, q2, q3]
    """
    arg = np.zeros(4,1)
    you = np.zeros(4,1)
    ment= np.zeros(4,1)

    arg[0] = np.cos(roll/2)
    arg[3] = np.sin(roll/2)

    you[0] = np.cos(pitch/2)
    you[2] = np.sin(pitch/2)

    ment[0] = np.cos(yaw/2)
    ment[1] = np.sin(yaw/2)

    return np.cross(arg,np.cross(you,ment))

def quaternion_to_euler(q):
    """
    Transformation between quaternion to Euler angles (roll, pitch, yaw)

    :param q: Quaternion vector [q0, q1, q2, q3]
    :return: Euler angle vector [roll, pitch, yaw]
    """
    q0, q1, q2, q3 = q

    # Roll (x-axis)
    roll = np.atan2(
        2*(q0*q1 + q2*q3),
        1 - 2*(q1**2 + q2**2)
    )

    # Pitch (y-axis)
    sinp = 2*(q0*q2 - q3*q1)
    pitch = np.arcsin(np.clip(sinp, -1, 1))  # avoid numerical issues

    # Yaw (z-axis)
    yaw = np.atan2(
        2*(q0*q3 + q1*q2),
        1 - 2*(q2**2 + q3**2)
    )

    return np.array([roll, pitch, yaw])

def euler_to_dcm(roll, pitch, yaw):
    """
    Transformation between Euler angles (roll, pitch, yaw) to Direction Cosine Matrix

    :param roll: Roll angle [degrees]
    :param pitch: Pitch angle [degrees]
    :param yaw: Yaw angle [degrees]
    :return: DCM 3x3 matrix, R
    """
    return np.array([
        [np.cos(pitch)*np.cos(yaw),     np.sin(roll)*np.sin(pitch)*np.cos(yaw) - np.cos(roll)*np.sin(yaw),  np.cos(roll)*np.sin(pitch)*np.cos(yaw) + np.sin(roll)*np.sin(yaw)]             
        [np.cos(pitch)*np.sin(yaw),     np.sin(roll)*np.sin(pitch)*np.sin(yaw) + np.cos(roll)*np.cos(yaw),  np.cos(roll)*np.sin(pitch)*np.sin(yaw) - np.sin(roll)*np.cos(yaw)]             
        [-np.sin(pitch),                np.sin(roll)*np.cos(pitch),                                         np.cos(roll)*np.cos(pitch)]             
                     ])

def dcm_to_euler(R):
    """
    Transformation between Direction Cosine Matrix to Euler angles (roll, pitch, yaw)

    :param R: DCM 3x3 matrix
    :return: Euler angle vector [roll, pitch, yaw]
    """
    roll = np.atan2(R[1,2], R[2,2])
    pitch = np.asin(-R[0,2])
    yaw = np.atan2(R[0,1], R[0,0])

    return [roll, pitch, yaw]

# Quaternion helper functions

    #Quaternion conjugator
def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

    #Quaternion multiplicator
def quat_mult(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

###################################
# Assignment 6 | Algorithms       #
###################################

def parse_tle_exponential(field):
    """
    Parse TLE scientific notation fields.

    Example:
        ' 36248-4' -> 0.36248e-4
        '-12345-5' -> -0.12345e-5
    """

    field = field.strip()

    if not field:
        return 0.0

    sign = 1.0

    if field[0] == '-':
        sign = -1.0
        field = field[1:]

    elif field[0] == '+':
        field = field[1:]

    mantissa = field[:-2]
    exponent = field[-2:]

    return sign * float(f"0.{mantissa}e{exponent}")


def read_TLE_file(file_name, satellite_name=None):
    """
    Read TLE data from file.

    Parameters
    ----------
    file_name : str
        Path to TLE file.

    satellite_name : str or None
        Optional satellite name filter.

    Returns
    -------
    list of dict
        Parsed TLE entries.
    """

    tle_data = []

    with open(file_name, "r") as f:
        lines = [line.rstrip() for line in f]

    if len(lines) % 3 != 0:
        raise ValueError(
            "TLE file must contain groups of 3 lines."
        )

    for k in range(0, len(lines), 3):

        name = lines[k]
        line1 = lines[k + 1]
        line2 = lines[k + 2]

        # Basic validation
        if not line1.startswith("1"):
            continue

        if not line2.startswith("2"):
            continue

        # Optional satellite filter
        if satellite_name is not None:
            if satellite_name.lower() not in name.lower():
                continue

        # ----- LINE 1 -----

        epoch = float(line1[18:32])

        dn_bar = float(line1[33:43])

        ddn_bar = parse_tle_exponential(
            line1[44:52]
        )

        bstar = parse_tle_exponential(
            line1[53:61]
        )

        # ----- LINE 2 -----

        inclination = float(line2[8:16])

        raan = float(line2[17:25])

        eccentricity = float(
            "0." + line2[26:33].strip()
        )

        arg_perigee = float(line2[34:42])

        mean_anomaly = float(line2[43:51])

        revs_per_day = float(line2[52:63])

        # ----- UNIT CONVERSIONS -----

        # Mean motion [rad/s]
        n = (
            2 * np.pi * revs_per_day
            / (24 * 3600)
        )

        # First derivative
        n_dot = (
            4 * np.pi * dn_bar
            / (24 * 3600)**2
        )

        # Second derivative
        n_ddot = (
            12 * np.pi * ddn_bar
            / (24 * 3600)**3
        )

        tle_entry = {

            "name": name,

            "epoch": epoch,

            "e": eccentricity,

            "i": np.radians(inclination),

            "Omega": np.radians(raan),

            "w": np.radians(arg_perigee),

            "M_e": np.radians(mean_anomaly),

            "revs_per_day": revs_per_day,

            "n": n,

            "n_dot": n_dot,

            "n_ddot": n_ddot,

            "bstar": bstar
        }

        tle_data.append(tle_entry)

    return tle_data