import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ase.units import kB
from scipy.spatial import cKDTree

class Spin_MonteCarlo_Simulator:
    def __init__(self, structure, temperature, 
                 J1, J2, J3, L, A, 
                 g, gamma, B_z, E_z,
                 orientations_lst, f_neighbor_array, s_neighbor_array, t_neighbor_array,
                 random_seed=None):
        """
        Initialize a Spin Monte Carlo simulator.

        Parameters:
        - structure (ase.Atoms): Atomic structure for the simulation (Positions of the structure must be all positive!).
        - temperature (float): Temperatures (in Kelvin) for Monte Carlo simulation.
        - J1 (float): The first NN isotropic exchange parameter in eV.
        - J2 (float): The second NN isotropic exchange parameter in eV.
        - J3 (float): The third NN isotropic exchange parameter in eV.
        - L (float): The anisotropic symmetric exchange parameter in eV.
        - A (float): The easy-axis single ion anisotropy parameter in eV.
        - g (float): Landé g-factor
        - gamma (float): Electric field coupling constant in e.Å (e is the electron charge in SI units)
        - B_z (float): Magnetic field strength in Tesla
        - E_z (float): Electric field strength in V/Å
        - orientations_lst (list of tuple): List of (theta, phi) pairs in degrees, representing the initial spin orientations.
        - f_neighbor_array (numpy.ndarray): Array of first NN neighbor indices for each site in the structure.
        - s_neighbor_array (numpy.ndarray): Array of second NN neighbor indices for each site in the structure.
        - t_neighbor_array (numpy.ndarray): Array of third NN neighbor indices for each site in the structure.
        """
        
        self.structure = structure
        self.n_spins = len(structure)
        self.temperature = temperature
        self.J1 = J1
        self.J2 = J2
        self.J3 = J3
        self.L = L
        self.A = A

        self.g = g          
        self.gamma = gamma  
        self.B_z = B_z      
        self.E_z = E_z      

        self.orientations_lst = orientations_lst
        self.f_neighbor_array = f_neighbor_array
        self.s_neighbor_array = s_neighbor_array
        self.t_neighbor_array = t_neighbor_array
        self.random_seed = random_seed
        

        # Create an initial spin-lattice array (an array of tuples (x, y, theta, phi))
        # Extract the atomic positions from the ASE structure
        positions = structure.get_positions()

        # Create a NumPy array with NaN values for the third and fourth entries
        spin_lattice = np.column_stack((positions[:, 0], positions[:, 1], 
                                       np.nan * np.ones(positions.shape[0]), 
                                       np.nan * np.ones(positions.shape[0])))

        self.spin_lattice = spin_lattice
        
        
        # Remove old folders if they exist
        if os.path.exists("energies"):
            shutil.rmtree("energies")
        if os.path.exists("spin_configs"):
            shutil.rmtree("spin_configs")  
            
        # Create folders for initial and final spin configurations
        os.makedirs("energies")
        os.makedirs("spin_configs")    
        

    def generate_random_spin_configuration(self):
        """
        Generate a random spin configuration with angles in degrees.
        Returns:
            np.ndarray: Random spin configuration.
        """
        np.random.seed(self.random_seed)

        # Randomly select pairs of (theta, phi) from the orientations_lst
        selected_indices = np.random.choice(np.arange(len(self.orientations_lst)), 
                                            size=self.spin_lattice.shape[0], replace=True)
        selected_pairs = self.orientations_lst[selected_indices]
    
        theta_values = selected_pairs[:, 0]
        phi_values = selected_pairs[:, 1]
        decorated_spin_lattice = np.copy(self.spin_lattice)
        decorated_spin_lattice[:, 2] = theta_values
        decorated_spin_lattice[:, 3] = phi_values
        return decorated_spin_lattice
                                            
    def generate_FM_spin_configuration(self, theta, phi):
        """
        Generate a ferromagnetic spin configuration along any axis.
        Returns:
            np.ndarray: Ferromagnetic spin configuration along specified axis.
        """
        size = self.spin_lattice.shape[0]
        theta_values = np.full(size, theta)
        phi_values = np.full(size, phi)
        decorated_spin_lattice = np.copy(self.spin_lattice)
        decorated_spin_lattice[:, 2] = theta_values
        decorated_spin_lattice[:, 3] = phi_values
        return decorated_spin_lattice                                        

    def calculate_site_energies(self, proposed_spin_config):
        """
        Calculate the energy of a spin configuration.
        Args:
            proposed_spin_config (np.ndarray): Spin configuration.
        Returns:
            float: Total energy of the configuration.
        """
        theta = np.radians(proposed_spin_config[:, 2])
        phi = np.radians(proposed_spin_config[:, 3])
  

        sin_theta_i = np.sin(theta)[:, np.newaxis]
        sin_theta_j = np.sin(theta)[self.f_neighbor_array]
        sin_theta_k = np.sin(theta)[self.s_neighbor_array]
        sin_theta_l = np.sin(theta)[self.t_neighbor_array]
                 
        cos_theta_i = np.cos(theta)[:, np.newaxis]
        cos_theta_j = np.cos(theta)[self.f_neighbor_array]
        cos_theta_k = np.cos(theta)[self.s_neighbor_array]
        cos_theta_l = np.cos(theta)[self.t_neighbor_array]
        
        cos_phi_ij = np.cos(phi[:, np.newaxis] - phi[self.f_neighbor_array])
        cos_phi_ik = np.cos(phi[:, np.newaxis] - phi[self.s_neighbor_array])
        cos_phi_il = np.cos(phi[:, np.newaxis] - phi[self.t_neighbor_array])
        
        # Calculate the isotropic exchange energy term for each spin
        f_iso_ex_energy = (-self.J1 / 2) * np.sum(sin_theta_i * sin_theta_j * cos_phi_ij + cos_theta_i * cos_theta_j, axis=1)
        s_iso_ex_energy = (-self.J2 / 2) * np.sum(sin_theta_i * sin_theta_k * cos_phi_ik + cos_theta_i * cos_theta_k, axis=1)
        t_iso_ex_energy = (-self.J3 / 2) * np.sum(sin_theta_i * sin_theta_l * cos_phi_il + cos_theta_i * cos_theta_l, axis=1)

        # Calculate the anisotropic symmetric exchange energy term for each spin
        aniso_symm_ex_energy =  (-self.L / 2) * np.sum(cos_theta_i*cos_theta_j, axis=1)      
        # Calculate the easy-axis single ion anisotropy energy term for each spin
        EA_single_ion_aniso_energy =  (-self.A / 1) * np.squeeze(cos_theta_i**2)


        # **Add the Zeeman and Electric Field Energy Terms**
        mu_B = 5.78838e-5  # Bohr magneton in eV/Tesla
        # Zeeman Energy Term
        E_Zeeman = -self.g * mu_B * self.B_z * np.squeeze(cos_theta_i)  # Resulting E_Zeeman in eV

        # Electric Field Energy Term
        E_Electric = -self.gamma * self.E_z * np.squeeze(cos_theta_i)   # Resulting E_Electric in eV

   
        # Calculate the total site energy for each spin
        site_energies = (f_iso_ex_energy + s_iso_ex_energy + t_iso_ex_energy) + aniso_symm_ex_energy + EA_single_ion_aniso_energy + E_Zeeman + E_Electric

        return site_energies
                                            

    def monte_carlo_simulation(self, initial_spin_config, sampling_sweep, sample_size, sampling_interval):
        """
        Perform Monte Carlo simulation.
        Args:
            initial_spin_config (np.ndarray): Initial spin configuration.
            sampling_sweep (int): Sweep at which to start sampling.
            sample_size (int): Number of samples to collect.
            sampling_interval (int): Number of sweeps between two successive samples.
        """
        np.random.seed(self.random_seed)
        
        sampled_energies = []
        gs_energies = []

        ensemble_energies = []
        ensemble_magnetizations = []
        spin_config_min = initial_spin_config
        
        num_sweeps = int(sampling_sweep + sample_size * sampling_interval)

        # Initialize tqdm for progress tracking
        progress_bar = tqdm(total=num_sweeps, desc='MC Sweeps', unit=' sweep', position=0, leave=True)


        sampling_count = 0

        for sweep in range(1, num_sweeps+1):
            for spin_idx in range(len(self.structure)):
                
                if (sweep ==1 and spin_idx ==0):
                    # Generate initial spin config
                    proposed_spin_config = initial_spin_config
                    # calculate the energy for the initial spin_config 
                    energy_new = np.sum(self.calculate_site_energies(proposed_spin_config))
                    # Accept the new spin configuration and its energy
                    spin_config = proposed_spin_config
                    energy = energy_new
                    # Gather info for initial spin config       
                    sampled_energies.append(energy_new)    
                    gs_energies.append(energy)

                else:
                    # Copy the up-to-date spin config
                    proposed_spin_config = np.copy(spin_config)

                    # Get a random index from the orientations list
                    random_index = np.random.randint(0, len(self.orientations_lst), size=1)

                    # Make an update 
                    proposed_spin_config[spin_idx, 2:4] = self.orientations_lst[random_index]

                    # calculate the energy for the proposed_spin_config   
                    energy_new = np.sum(self.calculate_site_energies(proposed_spin_config))
                    # delta energy
                    delta_energy = energy_new - energy

                    # Decide whether to accept or reject the spin flip based on the Metropolis criterion
                    if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / (kB * self.temperature)):
                        # Accept the new spin configuration and its energy
                        spin_config = proposed_spin_config
                        energy = energy_new

                    # Update the spin configuration corresponding to the minimum energy
                    if delta_energy < 0:
                        spin_config_min = proposed_spin_config

            # Gather info        
            sampled_energies.append(energy_new)    
            gs_energies.append(energy)


            if sweep >= sampling_sweep:
                if sampling_count % sampling_interval == 0:
                    # Sample energy
                    ensemble_energies.append(energy)
                    # Calculate the magnetization of the spin_config
                    m = calculate_normalized_magnetization(spin_config)
                    ensemble_magnetizations.append(m)

                    sampling_count += 1


            progress_bar.update(1)

        progress_bar.close()
        
        # Plot property convergence
        plot_energy_convergence(sampled_energies, gs_energies, self.temperature)
        plot_magnetization_convergence([np.linalg.norm(m) for m in ensemble_magnetizations], self.temperature)
        
        # Save
        np.save(f'energies/ensemble_energies_T{self.temperature}.npy', ensemble_energies)
        np.save(f'spin_configs/ensemble_magnetizations_T{self.temperature}.npy', ensemble_magnetizations)
        np.save(f'spin_configs/min_spin_config_T{self.temperature}.npy', spin_config_min)
        

##########  
##########                                              
                                            
def find_nearest_neighbors(structure, site_index, cutoff_distance):
    
    # Extract atomic positions and IDs
    atomic_positions = structure.get_positions()
    atomic_ids = list(range(len(atomic_positions)))

    # Build a KD tree from the atomic positions
    kd_tree = cKDTree(atomic_positions, boxsize=structure.cell.cellpar()[0:3])

    # Query the KD tree to find the nearest neighbors
    f_nn_indices= kd_tree.query_ball_point(atomic_positions[site_index][:], cutoff_distance)
    fs_nn_indices= kd_tree.query_ball_point(atomic_positions[site_index][:], 3**.5 * cutoff_distance)
    s_nn_indices= np.setdiff1d(fs_nn_indices, f_nn_indices)
    fst_nn_indices= kd_tree.query_ball_point(atomic_positions[site_index][:], 2 * cutoff_distance)
    t_nn_indices= np.setdiff1d(fst_nn_indices, fs_nn_indices)

    # Map the indices back to the original ASE structure
    f_nn_indices = (np.array(atomic_ids)[f_nn_indices]).tolist()
    s_nn_indices = (np.array(atomic_ids)[s_nn_indices]).tolist()
    t_nn_indices = (np.array(atomic_ids)[t_nn_indices]).tolist()
    
    f_nn_indices.remove(site_index)

    return f_nn_indices, s_nn_indices, t_nn_indices                                            
   
##########                                             
                                            
def calculate_normalized_magnetization(spin_config):
    """
    Calculate the magnetization of a spin configuration.
    Args:
        spin_config (np.ndarray): Spin configuration.
    Returns:
        np.ndarray: Magnetization vector.
    """
    # Extract x, y, theta, and phi from the input spin configuration
    x = spin_config[:, 0]
    y = spin_config[:, 1]
    theta = np.radians(spin_config[:, 2])
    phi = np.radians(spin_config[:, 3])

    # Calculate the individual spin vectors
    Sx = np.sin(theta) * np.cos(phi)
    Sy = np.sin(theta) * np.sin(phi)
    Sz = np.cos(theta)

    # Calculate the normalized magnetization 
    magnetization = np.array([np.mean(Sx), np.mean(Sy), np.mean(Sz)])

    return magnetization                                            
                                                                                                                               
##########     

def plot_energy_convergence(sampled_energies, gs_energies, temperature):
        # Plot energy curve
        # Create a list of step numbers for the x-axis
        steps = list(range(len(sampled_energies)))
        # Plot the sampled energies and ground state energies
        plt.figure(figsize=(8, 6), dpi=200)
        plt.plot(steps, sampled_energies, label='Sampled Energies', color='blue', alpha=0.4)
        plt.plot(steps, gs_energies, label='Ground State Energies', color='red', alpha=0.4)
        plt.xlabel('MC Sweeps', fontsize=14)
        plt.ylabel('Energy (eV)', fontsize=14)
        plt.title(f'Monte Carlo Simulation (Temperature = {temperature} K)', fontsize=17)
        plt.legend()
        plt.grid(True)
        # Save the energy plot
        plt.savefig(f'Energy_{temperature}.png')
        
def plot_magnetization_convergence(ensemble_magnetizations, temperature):
        # Plot magnetization curve
        # Create a list of step numbers for the x-axis
        steps = list(range(len(ensemble_magnetizations)))
        # Plot the ensemble magnetizations
        plt.figure(figsize=(8, 6), dpi=200)
        plt.plot(steps, ensemble_magnetizations, color='blue', alpha=0.4)
        plt.xlabel('MC Sweeps', fontsize=14)
        plt.ylabel('Normalized Magnetization', fontsize=14)
        plt.title(f'Monte Carlo Simulation (Temperature = {temperature} K)', fontsize=17)
        plt.grid(True)
        plt.ylim(0, 1)  
        # Save the magnetization plot
        plt.savefig(f'Magnetization_{temperature}.png')        
                                                   
