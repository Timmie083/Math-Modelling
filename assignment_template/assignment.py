import numpy as np
import simutils as su
import sat_lib as sl
import orbit_lib as ol
import simulator as sim
import plotter as pl
import math

class ScenarioAssignment(sim.BaseScenario):
    def __init__(self):
        self.RK4_x = None
        self.orbit_energy_plot = None
        self.m = None
        self.vi = None
        self.ri = None
        self.verlet_x_past = None
        self.verlet_x = None
        self.leapfrog_x = None
        self.q = None
        self.q_E = None
        self.euler_x = None
        self.pos_plot = None
        self.theta_E = None

    def init(self, t):

        RE = 6378.1363                  #[km]   Earth radius
        r31 = RE + 800                  #[km]   Radius used in assignment 3.1
        v31 = np.sqrt(ol.mu/(RE + 800)) #[km/s] Velocity used in assignment 3.1
        r32 = 7378                      #[km]   Radius used in assignment 3.2
        v32 = 9                         #[km/s] Velocity used in assignment 3.2

        # Satellite variables
        self.q = su.Quaternion()        # Satellite rotation
        self.ri = np.array([0, 0, 0])   # Satellite position NB! Check if correct per assignment
        self.vi = np.array([0, 0, 0]) #  Satellite velocity NB! Check if correct per assignment

        self.m = 8000 # Mass of the satellite [kg]

        v_temp = math.sqrt(ol.mu / (ol.R_E + 800))

        # Position from each integration method
        self.euler_x = np.concatenate([[ol.R_E + 800, 0, 0],[0, v_temp, 0]])
        self.leapfrog_x = np.concatenate([[ol.R_E + 800, 0, 0],[0, v_temp, 0]])

        self.verlet_x_past = None
        self.verlet_x = np.concatenate([[ol.R_E + 800, 0, 0],[0, v_temp, 0]])

        # Used for Assignment 3.2
        self.RK4_x = np.concatenate([self.ri, self.vi])

        # Earth rotation variables
        self.theta_E = 0 # Offset to the rotation
        temp = ol.polar2xyz(1, self.theta_E / 2) # Normalized XY from q_E
        self.q_E = su.Quaternion([temp[0], 0, 0, temp[1]])
        #self.q_E = su.Quaternion()

        # Data logging variables
        #self.pos_plot = np.concatenate(([t], self.ri)) # Initialize the plot data

        # Convert all energies to array
        #self.orbit_energy_plot = np.concatenate(([t], [ol.get_orbit_energy_state(self.euler_x, self.m), ol.get_orbit_energy_state(self.leapfrog_x, self.m), ol.get_orbit_energy_state(self.verlet_x, self.m)]))



    def update(self, t, dt):

        #Assignment 3.2
        k1 = 10e-4
        k2 = 10e-4
        ei = ol.get_orbit_eccentricity_vector_state(self.RK4_x)
        e = np.linalg.norm(ei)

        cos_theta = np.dot(ei, self.RK4_x[:3]) / (e * np.linalg.norm(self.RK4_x[:3]))
        ra = ol.get_orbit_apoapsis(self.RK4_x, e)
        rp = ol.get_orbit_periapsis(self.RK4_x, e)
        rc = ol.R_E + 1500

        T = (k1 * (rc - ra)) if cos_theta > 0.9 else (k2 * (rc - rp)) if cos_theta < -0.9 else 0

        ae = (T * self.RK4_x[3:] / np.linalg.norm(self.RK4_x[3:])) / self.m

        # Next step
        self.euler_x = su.step_euler(dt, t, self.euler_x, su.two_body)
        self.leapfrog_x = su.step_leapfrog(dt, t, self.leapfrog_x, su.two_body)
        self.verlet_x, self.verlet_x_past = su.step_verlet(dt, t, self.verlet_x, self.verlet_x_past, su.two_body), self.verlet_x
        self.RK4_x = su.step_RK4(dt, t, self.RK4_x, su.two_body, ae=ae)
        #self.ri = self.euler_x[:3]     # Get position vector NB! Check that numerical solver is correct
        #self.ri = self.leapfrog_x[:3]  # Get position vector NB! Check that numerical solver is correct
        #self.ri = self.verlet_x[:3]    # Get position vector NB! Check that numerical solver is correct
        #self.ri = self.RK4_x[:3]       # Get position vector NB! Check that numerical solver is correct

        # Calculate earth's rotation from time step
        self.theta_E += dt * ol.w_E
        temp = ol.polar2xyz(1, self.theta_E / 2) # Normalized XY from q_E
        self.q_E = su.Quaternion([temp[0], 0, 0, temp[1]])

        # Log orbit data
        #self.pos_plot = np.vstack((self.pos_plot, np.concatenate(([t], self.ri))))
        #self.orbit_energy_plot = np.vstack((self.orbit_energy_plot, np.concatenate(([t], [ol.get_orbit_energy_state(self.euler_x, self.m), ol.get_orbit_energy_state(self.leapfrog_x, self.m), ol.get_orbit_energy_state(self.verlet_x, self.m)]))))

    def get(self):
        return [
            ['satellite', self.ri, self.q],
            ['body_frame', self.ri, self.q],
            ['earth', np.zeros(3), self.q_E],
            ['ECEF frame', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()]]

    #def post_process(self, t, dt):
        # Plot orbit of satellite
        file = su.log_pos("assignment3_position", self.pos_plot)
        self.pos_plot = None # Clear the data after its saved
        pl.line_plot(file)

        file = su.log_pos("assignment3_energy", self.orbit_energy_plot)
        self.orbit_energy_plot = None  # Clear the data after its saved
        pl.line_plot(file, labels=["Euler", "Leapfrog", "Verlet"])

def main():

    sim_config = {
        't_0': 0,
        't_e': 56000,
        't_step': 100,
        'speed_factor': 100,
        'anim_dt': 0.04,
        'scale_factor': 1000,
        'visualise': True
    }

    scenario = ScenarioAssignment()
    sim.create_and_start_simulation(sim_config, scenario)

if __name__ == "__main__":
    main()

