import numpy as np
import simutils as su
import sat_lib as sl
import orbit_lib as ol
import simulator as sim

# --- TLE epoch ---
epoch = 26097.16668981  # from your TLE

JD0 = ol.epoch_to_julian_date(epoch)
theta0 = ol.sidereal_angle(JD0)

RE = 6378.1363;
my = 398604.415;

r0 = np.array([RE + 800, 0, 0]);
v0 = np.array([0, 0, np.sqrt(my / np.linalg.norm(r0))]);

x0 = np.hstack((r0, v0))  

r = RE + 400;
t_0 = 0;
t_step = 1;

T = ((2*np.pi)/np.sqrt(my))*r**(3/2);

t_e = T;
speed_factor = 100;
anim_dt = 0.04;
scale_factor = 1000;

theta = 0;
omega_E = 7.292115e-5;

r_i = r*(np.array([np.cos(theta), 
                  np.sin(theta), 
                  0]));

theta_dot = 2*np.pi/T;

def f(t, x):
    r = x[:3]
    v = x[3:]
    
    r_norm = np.linalg.norm(r)
    a = -my * r / r_norm**3
    
    return np.hstack((v, a))

class ScenarioAssignment3(sim.BaseScenario):
    def init(self, t):

        self.ri = np.array([7378, 0, 0]) # Satellite position
        self.vi = np.array([0, 0, 9]) # Satellite velocity

        self.RK4_x = np.concatenate([self.ri, self.vi])

        self.m = 8000
        self.x = np.hstack((r0, v0))  # x0

        a0 = f(t, self.x)[3:]
        r1 = r0 + v0 * t_step + 0.5 * a0 * t_step**2
        v1 = v0  

        self.x_prev = self.x.copy()              
        self.x = np.hstack((r1, v1))

        # Visualization
        self.q = su.Quaternion()
        self.q_E = su.Quaternion()
        self.r_ecef = self.x[:3]

    def update(self, t, dt):
        # --- NUMERICAL STEP ---
        #self.x = su.step_euler(dt, t, self.x, f)
        #self.x = su.step_leapfrog(dt, t, self.x, f)
        #self.x = su.step_verlet(dt, t, self.x, self.x_prev, f)

        # Assignment 3.2 (credit til Askar for mega hjelp)

        k1 = 10e-4
        k2 = 10e-4
        ei = ol.get_orbit_eccentricity_vector_state(self.RK4_x, my)
        e = np.linalg.norm(ei)

        cos_theta = np.dot(ei, self.RK4_x[:3]) / (e * np.linalg.norm(self.RK4_x[:3]))
        ra = ol.get_orbit_apoapsis(self.RK4_x, e, my)
        rp = ol.get_orbit_periapsis(self.RK4_x, e, my)
        rc = ol.R_E + 1500
        
        T = (k1 * (rc-ra)) if cos_theta > 0.9 else (k2 * (rc-rp)) if cos_theta < -0.9 else 0
        
        ae = (T * self.RK4_x[3:] / np.linalg.norm(self.RK4_x[3:])) / self.m

        self.RK4_x = su.step_RK4(dt, t, self.RK4_x, su.two_body,ae)
        self.ri = self.RK4_x[:3]
        self.vi = self.RK4_x[3:]

        # Earth rotation
        theta_E = ol.earth_rotation_angle(theta0, t)

        R_E = np.array([
            [np.cos(theta_E),  np.sin(theta_E), 0],
            [-np.sin(theta_E), np.cos(theta_E), 0],
            [0,                0,               1]
        ])


        #r_i = self.x[:3]
        #self.r_ecef = R_E @ r_i

        self.q_E = su.Quaternion([
            np.cos(theta_E / 2),
            0,
            0,
            np.sin(theta_E / 2)
        ])

    def get(self):
        return [
            ['satellite', self.ri, self.q],
            ['earth', np.zeros(3), self.q_E],
            ['ECEF frame', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()]
        ]

    def post_process(self, t, dt):
        r = self.x[:3]
        point = np.array([t, r[0], r[1], r[2]])

        if not hasattr(self, "pos_plot"):
            self.pos_plot = np.array([point])
        else:
            self.pos_plot = np.vstack((self.pos_plot, point))

        if t >= 20000 - dt:
            su.log_pos('assignment3_position', self.pos_plot)

def main():

    

    sim_config = {
        't_0': 0,
        't_e': 53000,
        't_step': 100,
        'speed_factor': 1,
        'anim_dt': 1/25,
        'scale_factor': 1000,
        'visualise': True
    }

    scenario = ScenarioAssignment3()
    sim.create_and_start_simulation(sim_config, scenario)



if __name__ == "__main__":
    main()

