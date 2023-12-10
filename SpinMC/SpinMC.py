import os
import shutil
import io
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from tqdm import tqdm
from ase.io import write, read
from ase.build import sort, make_supercell
from ase.units import kB
from scipy.spatial import cKDTree

class Spin_MonteCarlo_Simulator:
    def __init__(self, structure, temperatures, J, A, max_spin_group,
                 orientations_lst, neighbor_array):
        """
        Initialize a Spin Monte Carlo simulator.

        Parameters:
        - structure (ase.Atoms): Atomic structure for the simulation (Positions of the structure must be all positive!).
        - temperatures (list of float): List of temperatures (in Kelvin) for Monte Carlo simulations.
        - J (float): Direct exchange parameter in eV.
        - A (float): Anisotropy parameter in eV.
        - max_spin_group (int): Maximum number of spins to update in a Monte Carlo step (minimum is one).
        - orientations_lst (list of tuple): List of (theta, phi) pairs in degrees.
        - neighbor_array (numpy.ndarray): Array of neighbor indices for each site in the structure.
        """
        
        self.structure = structure
        self.n_spins = len(structure)
        self.temperatures = temperatures
        self.J = J
        self.A = A
        self.max_spin_group = max_spin_group
        self.orientations_lst = orientations_lst
        self.neighbor_array = neighbor_array

        # Create an initial spin-lattice array (an array of tuples (x, y, theta, phi))
        # Extract the atomic positions from the ASE structure
        positions = structure.get_positions()

        # Create a NumPy array with NaN values for the third and fourth entries
        spin_lattice = np.column_stack((positions[:, 0], positions[:, 1], 
                                       np.nan * np.ones(positions.shape[0]), 
                                       np.nan * np.ones(positions.shape[0])))

        self.spin_lattice = spin_lattice
        
        
        # Remove old folders if they exist
        if os.path.exists("initial_spin_configs"):
            shutil.rmtree("initial_spin_configs")
        if os.path.exists("final_spin_configs"):
            shutil.rmtree("final_spin_configs")
        if os.path.exists("Energy_plots"):
            shutil.rmtree("Energy_plots")

        # Create folders for initial and final spin configurations
        os.makedirs("initial_spin_configs")
        os.makedirs("final_spin_configs")
        os.makedirs("Energy_plots")
        

    def generate_random_spin_configuration(self, random_seed=None):
        """
        Generate a random spin configuration with angles in degrees.
        Args:
            random_seed (int): Random seed for reproducibility.
        Returns:
            np.ndarray: Random spin configuration.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

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
                                            
    def generate_FM_spin_configuration(self):
        """
        Generate a ferromagnetic spin configuration along x axis.
        Returns:
            np.ndarray: Ferromagnetic spin configuration along x axis.
        """
        size = self.spin_lattice.shape[0]
        theta_values = np.full(size, spin_theta)
        phi_values = np.full(size, spin_phi)
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

        # Compute dot products of spins using vectorized operations
        cos_theta = np.cos(theta)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        sin_phi = np.sin(phi)

        # Calculate the dot products for each spin and its neighbors
        dot_products = (cos_theta[:, np.newaxis] \
                        * cos_theta[self.neighbor_array]) + \
                       (sin_theta[:, np.newaxis] \
                        * cos_phi[:, np.newaxis] * sin_theta[self.neighbor_array] * cos_phi[self.neighbor_array]) + \
                       (sin_theta[:, np.newaxis] \
                        * sin_phi[:, np.newaxis] * sin_theta[self.neighbor_array] * sin_phi[self.neighbor_array])

        # Initialize an array for the anisotropy energy terms
        anisotropy_energy_terms = np.zeros(len(proposed_spin_config))
        # Identify spins with theta = 90 (in-plane) and set anisotropy_energy_term to 0
        in_plane_spins = (theta == np.radians(90))
        anisotropy_energy_terms[in_plane_spins] = 0.0
        # Identify spins with theta = 0 or 180 (z-direction) and set anisotropy_energy_term to -A/4
        z_direction_spins = (theta == np.radians(0)) | (theta == np.radians(180))
        anisotropy_energy_terms[z_direction_spins] = -self.A / 4

        # Calculate the exchange energy term for each spin
        exchange_energy_terms = (-self.J / 8) * np.sum(dot_products, axis=1)

        # Calculate the total site energy for each spin
        site_energies = anisotropy_energy_terms + exchange_energy_terms

        return site_energies
                                            

    def monte_carlo_simulation(self, temperature, num_steps, initial_spin_config, 
                               sampling_step, sample_size, sampling_interval):
        """
        Perform Monte Carlo simulation.
        Args:
            temperature (float): Temperature for the simulation.
            num_steps (int): Total number of simulation steps.
            initial_spin_config (np.ndarray): Initial spin configuration.
            sampling_step (int): Step at which to start sampling.
            sample_size (int): Number of samples to collect.
            sampling_interval (int): Interval between two successive samples.
        Returns:
            np.ndarray: Final spin configuration.
        """
        sampled_energies = []
        gs_energies = []

        ensemble_energies = []
        ensemble_spin_configs = []

        # Initialize tqdm for progress tracking
        progress_bar = tqdm(total=num_steps, desc='MC Steps', unit=' step', position=0, leave=True)


        sampling_count = 0

        for step in range(num_steps):
            if step ==0:
                # Generate initial spin config
                proposed_spin_config = initial_spin_config
                # calculate the energy for the initial spin_config 
                energy_new = np.sum(self.calculate_site_energies(proposed_spin_config))
                # Accept the new spin configuration and its energy
                spin_config = proposed_spin_config
                energy = energy_new

            else:
                # Copy the up-to-date spin config
                proposed_spin_config = np.copy(spin_config)

                # select a random number of spins to change their states
                n_group = random.choice(np.arange(1, 1+self.max_spin_group))

                # Randomly choose lattice sites (of size n_group)
                site_indices = random.sample(range(len(self.structure)), n_group)

                # Change the spin orientations
                for i in site_indices:
                    # Get the old spin orientation
                    theta = proposed_spin_config[i][2]; phi = proposed_spin_config[i][3]
                    # Initialize the new theta and phi
                    theta_proposed = theta; phi_proposed = phi
                    # Pick new theta and phi
                    while (theta_proposed == theta and phi_proposed == phi):
                        # select a random spin from orientations_lst
                        spin_orientation = random.choice(self.orientations_lst)
                        theta_proposed = spin_orientation[0]
                        phi_proposed = spin_orientation[1]
                    # update with the new spin 
                    proposed_spin_config[i][2] =  theta_proposed 
                    proposed_spin_config[i][3] =  phi_proposed


                # calculate the energy for the proposed_spin_config   
                energy_new = np.sum(self.calculate_site_energies(proposed_spin_config))
                # delta energy
                delta_energy = energy_new - energy

                # Decide whether to accept or reject the spin flip based on the Metropolis criterion
                if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / (kB * temperature)):
                    # Accept the new spin configuration and its energy
                    spin_config = proposed_spin_config
                    energy = energy_new

            # Gather info        
            sampled_energies.append(energy_new)    
            gs_energies.append(energy)      

            progress_bar.update(1)

            if step >= sampling_step:
                if sampling_count % sampling_interval == 0:
                    ensemble_spin_configs.append(spin_config)
                    ensemble_energies.append(energy)
                sampling_count += 1

        progress_bar.close()

        # Plots

        # Create a list of step numbers for the x-axis
        steps = list(range(num_steps))
        # Plot the sampled energies and ground state energies
        plt.figure(figsize=(8, 6), dpi=200)
        plt.plot(steps, sampled_energies, label='Sampled Energies', color='blue', alpha=0.7)
        plt.plot(steps, gs_energies, label='Ground State Energies', color='red', alpha=0.7)
        plt.xlabel('MC Steps', fontsize=14)
        plt.ylabel('Energy (eV)', fontsize=14)
        plt.title(f'Monte Carlo Simulation (Temperature = {temperature} K)', fontsize=17)
        plt.legend()
        plt.grid(True)
        # Save the energy plot
        energy_plot_path = os.path.join("Energy_plots", 'Energy_{}.png'.format(temperature))
        plt.savefig(energy_plot_path)
                                            
        # Plot the initial spin plot
        plot_spin_configuration(initial_spin_config, self.calculate_site_energies(initial_spin_config), temperature, 'i')
        # Plot the final spin plot
        plot_spin_configuration(spin_config, self.calculate_site_energies(spin_config), temperature, 'f')

        return spin_config, ensemble_spin_configs, ensemble_energies
                                           
                                                                                    
                                            
    def run_simulation_range(self, sampling_step, sample_size, sampling_interval):
        """
        Perform Monte Carlo simulations over a range of temperatures, 
        calculate specific heat, magnetization, and square magnetization.
        Args:
            sampling_step (int): Step at which to start sampling.
            sample_size (int): Number of samples to collect.
            sampling_interval (int): Interval between two successive samples.
            temperatures: array of temperatures
        """
        # Reverse the temperatures
        reversed_temperatures = np.flip(self.temperatures)

        num_steps = int(sampling_step + sample_size * sampling_interval)

        # Generate initial spin config
        gs_spin_config = self.generate_random_spin_configuration(random_seed=None)


        # Create lists to store calculated values for specific heat, magnetization, and square magnetization
        temperature_values = []
        specific_heat_values = []
        magnetization_values = []
        square_magnetization_values = []
                                            
                                            
        # Iterate over temperatures
        for temperature in reversed_temperatures:

            initial_spin_config = gs_spin_config
                 
            # Perform a Monte Carlo simulation                                
            gs_spin_config, ensemble_spin_configs, ensemble_energies = \
                                            self.monte_carlo_simulation(temperature, num_steps, initial_spin_config,
                                                                        sampling_step, sample_size, sampling_interval)                                

            temperature_values.append(temperature)

            # Calculate specific heat
            specific_heat = calculate_specific_heat(ensemble_energies, temperature, self.n_spins)
            specific_heat_values.append(specific_heat)

            # Calculate expected magnetization and square magnetization
            expected_magnetization, expected_square_magnetization = \
            calculate_expected_magnetization_and_square_magnetization(ensemble_spin_configs)

            magnetization_values.append(np.linalg.norm(expected_magnetization))
            square_magnetization_values.append(np.linalg.norm(expected_square_magnetization))
                                            
                                            
            # Plot simulation results
            plot_simulation_results(temperature_values, 
                                    specific_heat_values,
                                    magnetization_values, square_magnetization_values)                                
                                            
##########  
##########  
##########  
##########                                              
                                            
def find_nearest_neighbors(structure, site_index, cutoff_distance):
    
    # Extract atomic positions and IDs
    atomic_positions = structure.get_positions()
    atomic_ids = list(range(len(atomic_positions)))

    # Build a KD tree from the atomic positions
    kd_tree = cKDTree(atomic_positions, boxsize=structure.cell.cellpar()[0:3])

    # Query the KD tree to find the nearest neighbors
    nearest_neighbor_indices = kd_tree.query_ball_point(atomic_positions[site_index][:], cutoff_distance)

    # Map the indices back to the original ASE structure
    nearest_neighbor_indices = [atomic_ids[i] for i in nearest_neighbor_indices]
    
    nearest_neighbor_indices.remove(site_index)

    return nearest_neighbor_indices                                            
   
##########                                             
                                            
def calculate_magnetization(spin_config):
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

    # Calculate the magnetization (net magnetic moment)
    magnetization = np.array([np.sum(Sx), np.sum(Sy), np.sum(Sz)])

    return magnetization                                            
                                          
##########    
                                            
def calculate_expected_magnetization_and_square_magnetization(spin_configs):
    """
    Calculate expected magnetization and square magnetization from a list of spin configurations.
    Args:
        spin_configs (List[np.ndarray]): List of spin configurations.
    Returns:
        np.ndarray: Expected magnetization.
        np.ndarray: Expected square magnetization.
    """
    n_spins = spin_configs[0].shape[0]
                                            
    total_magnetization = np.zeros(3)
    total_square_magnetization = np.zeros(3)  

    for spin_config in spin_configs:
        magnetization = calculate_magnetization(spin_config)
        total_magnetization += magnetization
        total_square_magnetization += np.square(magnetization)

    # Calculate the expected values of magnetization and square_magnetization
    expected_magnetization = total_magnetization / (len(spin_configs) * n_spins)
    expected_square_magnetization = total_square_magnetization / (len(spin_configs) * n_spins**2)

    return expected_magnetization, expected_square_magnetization

########## 
                                            
def calculate_specific_heat(energies, temperature, n_spins):
    """
    Calculate specific heat from a list of energies and temperature.
    Args:
        energies (List[float]): List of energies.
        temperature (float): Temperature.
        n_spins (int): Number of spins.
    Returns:
        float: Specific heat.
    """
    # Calculate the specific heat using the provided energy values and temperature
    expected_energy = np.mean(energies)
    expected_square_energy = np.mean(np.square(energies))
    specific_heat = (expected_square_energy - expected_energy ** 2) / (n_spins * (kB* temperature) ** 2)

    return specific_heat                                            
              
##########    
                                            
def calculate_spin_spin_correlation(spin_config):
    """
    Calculate the spin-spin correlation function for a given spin configuration.

    Args:
        spin_config (np.ndarray): Spin configuration with shape (N, 4) where N is the number of spins.

    Returns:
        np.ndarray: Spin-spin correlation function, which is an array of values for different distances.
    """
    N = len(spin_config)
    max_distance = 10  # You can adjust this based on your system's size
    correlation_function = np.zeros(max_distance)

    for i in range(N):
        for j in range(i + 1, N):
            distance = np.linalg.norm(spin_config[i, :2] - spin_config[j, :2])
            if distance < max_distance:
                spin_i = np.array([np.sin(np.radians(spin_config[i, 2])) * np.cos(np.radians(spin_config[i, 3])),
                                   np.sin(np.radians(spin_config[i, 2])) * np.sin(np.radians(spin_config[i, 3])),
                                   np.cos(np.radians(spin_config[i, 2]))])
                spin_j = np.array([np.sin(np.radians(spin_config[j, 2])) * np.cos(np.radians(spin_config[j, 3])),
                                   np.sin(np.radians(spin_config[j, 2])) * np.sin(np.radians(spin_config[j, 3])),
                                   np.cos(np.radians(spin_config[j, 2]))])

                correlation_function[int(distance)] += np.dot(spin_i, spin_j)

    return correlation_function                                            
 
##########    

def plot_spin_configuration(spin_config, site_energies, temperature, s):
    """
    Plot a spin configuration with colored spin arrows based on site energy.
    Args:
        spin_config (np.ndarray): Spin configuration.
        site_energies (np.ndarray): Site energies.
        temperature (float): Temperature.
        s (str): Identifier for the spin state, which can be "i" (initial) or "f" (final).
    """
    # Extract x and y coordinates from spin_config
    x = spin_config[:, 0]
    y = spin_config[:, 1]

    # Extract theta and phi values in degrees
    theta_degrees = spin_config[:, 2]
    phi_degrees = spin_config[:, 3]

    # Convert degrees to radians
    theta = np.radians(theta_degrees)
    phi = np.radians(phi_degrees)

    # Create a scatter plot for spins pointing up and down
    plt.figure(figsize=(8, 8), dpi=500)

    # Plot spins pointing up (theta = 0 degrees)
    plt.scatter(x[theta_degrees == 0], y[theta_degrees == 0], 
                c='blue', marker='o', label='Spin Up', s=7)

    # Plot spins pointing down (theta = 180 degrees)
    plt.scatter(x[theta_degrees == 180], y[theta_degrees == 180], 
                c='red', marker='x', label='Spin Down', s=7)
    
    # For spins in the plane (0 < theta < 180 degrees), use quiver to plot arrows in the phi direction
    in_plane_indices = (0 < theta_degrees) & (theta_degrees < 180)
    x_in_plane = x[in_plane_indices]
    y_in_plane = y[in_plane_indices]
    phi_in_plane = phi[in_plane_indices]
    
    # Plot spins in-plane (theta = 90 degrees)
    #plt.scatter(x_in_plane, y_in_plane, c='black', marker='.', label='in-plane', s=7)

    # Calculate arrow components
    arrow_length = 1.0  # Adjust the arrow length as needed
    u = arrow_length * np.cos(phi_in_plane)
    v = arrow_length * np.sin(phi_in_plane)
    
    # Calculate arrow colors based on site energies
    arrow_colors = site_energies[in_plane_indices]

    # Define a colormap for the arrows
    arrow_cmap = cm.viridis  # You can choose a different colormap for the arrows

    # Plot arrows with colored lines and adjust linewidth
    plt.quiver(x_in_plane, y_in_plane, u, v, arrow_colors, cmap=arrow_cmap, pivot='mid', width=0.0013)

    plt.xlabel('x (Å)')
    plt.ylabel('y (Å)')
    plt.title(f'Spin Configuration (Temperature = {temperature} K)')
    plt.grid(True, linewidth=0.2)
    
    # Ensure equal aspect ratio to make arrow lengths consistent
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Create a color bar legend
    cbar = plt.colorbar(orientation='vertical', label='Site Energy (eV)', shrink=0.75)
    
    # Move the legend to the top side outside the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=2)
    
    # Save the figure
    if s == 'i':
        spin_config_path = 'initial_spin_configs'
    elif s == 'f':
        spin_config_path = 'final_spin_configs'
    file_name = f'spin_{temperature}_{s}.png'
    file_path = os.path.join(spin_config_path, file_name)
    plt.savefig(file_path)


##########                                             
                                                                                        
def plot_simulation_results(temperature_values, 
                            specific_heat_values,
                            magnetization_values, square_magnetization_values):
    """
    Plot specific heat, magnetization, square magnetization, number of vortices, and correlation function.
    """
    # Plot specific heat
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(temperature_values, specific_heat_values, marker='o', linestyle='-')
    plt.xlabel('Temperature (K)', fontsize=14)
    plt.ylabel('Specific Heat / kB', fontsize=14)
    plt.title('Specific Heat vs. Temperature', fontsize=17)
    plt.grid(True)
    plt.savefig('Specific Heat.png')

    # Plot magnetization
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(temperature_values, magnetization_values, marker='o', linestyle='-')
    plt.xlabel('Temperature (K)', fontsize=14)
    plt.ylabel('Normalized Magnetization', fontsize=14)
    plt.title('Magnetization vs. Temperature', fontsize=17)
    plt.grid(True)
    plt.savefig('Magnetization.png')

    # Plot square magnetization
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(temperature_values, square_magnetization_values, marker='o', linestyle='-')
    plt.xlabel('Temperature (K)', fontsize=14)
    plt.ylabel('Square Magnetization', fontsize=14)
    plt.title('Square Magnetization vs. Temperature', fontsize=17)
    plt.grid(True)
    plt.savefig('Square Magnetization.png')
      
##########                                              
