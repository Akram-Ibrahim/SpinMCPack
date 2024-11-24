import os, sys
import shutil


def create_simulation_folder(T, B_z, E_z):
    # Create a folder name that includes T, B_z, and E_z
    folder_name = f"T_{T}_B_{B_z}_E_{E_z}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def write_script_template(folder_name, T, sampling_sweep, B_z, E_z):
    # The template script content
    script_content = f"""
import numpy as np
from ase.io import read
from ase.build import sort, make_supercell
import SpinMC

# Input values
##################
n_cells = 40
temperature = {T}

J1 =; J2 =; J3= ; L = -0.57/1000; A = -0.77/1000 # in eV
g = ...           # Replace with actual g-factor
gamma = ...       # Replace with actual gamma in e·Å
B_z = {B_z}       # Magnetic field strength in Tesla
E_z = {E_z}       # Electric field strength in V/Å
angular_res = 0.5
cutoff_distance = 3.96124 + 0.1  # on-lattice Ni-Ni distance + skin distance 
sampling_sweep = {sampling_sweep}; sample_size = 2e4; sampling_interval = 1
model_type = '3D'                # Choose from Ising, XY, 3D
##################

# Read structure
##################
# Read primitive structure
prim_struc = read('POSCAR-conv')
# Make supercell
super_struc = make_supercell(prim_struc, np.array([[n_cells, 0, 0], [0, round(n_cells/3**.5), 0], [0, 0, 1]]))
##################

# **Avoid small negative positions from ASE to ensure adaptability with cKDTree**
# Clip the positions to be non-negative
non_negative_positions = np.clip(super_struc.positions, 0, None)

# Set the clipped positions back to the structure
super_struc.set_positions(non_negative_positions)

# occupation list
##################
if model_type == 'Ising':
    
    # Possible values for theta and phi
    theta_values = [0, 180] # +Z, -Z spins only
    phi = 0                 # phi value is set to 0 when theta = 0, 180

    # Create a 2D array with all combinations
    orientations_lst = []
 
    for theta in theta_values:
        orientations_lst.append([theta, phi])

    # Convert the list of combinations into a NumPy array
    orientations_lst = np.array(orientations_lst)
    
elif model_type == 'XY':
    
    # Possible values for theta and phi
    theta = 90 # XY spins only
    phi_values = np.arange(0, 360, angular_res)

    # Create a 2D array with all combinations
    orientations_lst = []
    
    for phi in phi_values:
        orientations_lst.append([theta, phi])

    # Convert the list of combinations into a NumPy array
    orientations_lst = np.array(orientations_lst)


elif model_type == '3D':
    # theta [0, pi] <--> v [0 , 1]
    # theta = cos^-1(2*v-1) (rad)
    # theta = (180 / np.pi) * cos^-1(2*v-1) (deg)

    # Calculate the resolution in the v variable based on the theta resolution
    n_divs = 180 / angular_res
    res_v = 1 / n_divs

    # Possible values for v
    v_values = np.arange(0, 1, res_v)

    # Possible values for theta and phi
    theta_values = (180 / np.pi) * np.arccos(2 * v_values - 1)
    phi_values = np.arange(0, 360, angular_res)

    # Create a list of orientations
    orientations_lst = []
    for theta in theta_values:
        for phi in phi_values:
            orientations_lst.append([theta, phi])

    # Convert the list of combinations into a NumPy array
    orientations_lst = np.array(orientations_lst)      
            
##################

# Get neighbors array
print('Getting the neighbors array ....')

f_neighbor_array = []
s_neighbor_array = []
t_neighbor_array = []

for i, at in enumerate(super_struc):
    f_neighbor_indices, s_neighbor_indices, t_neighbor_indices = SpinMC.find_nearest_neighbors(super_struc, i, cutoff_distance)
    f_neighbor_array.append(f_neighbor_indices)    
    s_neighbor_array.append(s_neighbor_indices)    
    t_neighbor_array.append(t_neighbor_indices) 

f_neighbor_array = np.array(f_neighbor_array)
s_neighbor_array = np.array(s_neighbor_array)
t_neighbor_array = np.array(t_neighbor_array)

print('Successfully got the neighbors array ....')
    



# Run Monte Carlo
##################
# Initiate MC simulator
mc_simulator = SpinMC.Spin_MonteCarlo_Simulator(super_struc, temperature, J1, J2, J3, L, A, g, gamma, B_z, E_z,
                                                orientations_lst, f_neighbor_array, s_neighbor_array, t_neighbor_array, random_seed=42)

# Generate initial spin config
initial_spin_config = mc_simulator.generate_random_spin_configuration()

# Run
mc_simulator.monte_carlo_simulation(initial_spin_config, sampling_sweep, sample_size, sampling_interval)
##################

print("Finished successfully ...")
"""

    script_file_path = os.path.join(folder_name, 'script.py')
    with open(script_file_path, 'w') as script_file:
        script_file.write(script_content)

def copy_common_files(folder_name):
    shutil.copy('SpinMC.py', folder_name)
    shutil.copy('submission-script', folder_name)
    # Ensure 'POSCAR-conv' is copied to the folder
    shutil.copy('POSCAR-conv', folder_name)

def run_sbatch(folder_name):
    os.chdir(folder_name)
    os.system("sbatch submission-script")
    os.chdir("..")

def main(T, sampling_sweep, B_z, E_z):
    folder_name = create_simulation_folder(T, B_z, E_z)
    write_script_template(folder_name, T, sampling_sweep, B_z, E_z)
    copy_common_files(folder_name)
    run_sbatch(folder_name)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python template.py <temperature> <sampling_sweep> <B_z> <E_z>")
        sys.exit(1)

    try:
        T_input = float(sys.argv[1])
    except ValueError:
        print("Error: Temperature must be a number.")
        sys.exit(1)

    try:
        sampling_sweep_input = float(sys.argv[2])
    except ValueError:
        print("Error: Sampling sweep must be a number.")
        sys.exit(1)

    try:
        B_z_input = float(sys.argv[3])
    except ValueError:
        print("Error: B_z must be a number (magnetic field in Tesla).")
        sys.exit(1)

    try:
        E_z_input = float(sys.argv[4])
    except ValueError:
        print("Error: E_z must be a number (electric field in V/Å).")
        sys.exit(1)

    main(T_input, sampling_sweep_input, B_z_input, E_z_input)  
  
