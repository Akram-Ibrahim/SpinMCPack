import os, sys
import shutil


def create_simulation_folder(T):
    folder_name = str(T)
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def write_script_template(folder_name, T, sampling_sweep):
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

J = 71/1000; L = -0.57/1000; A = -0.77/1000 # in eV 
angular_res = 0.5
cutoff_distance = 3.41444 + 0.1  # on-lattice V-V distance + skin distance 
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

def run_sbatch(folder_name):
    os.chdir(folder_name)
    os.system("sbatch submission-script")
    os.chdir("..")

def main(T, sampling_sweep):
    folder_name = create_simulation_folder(T)
    write_script_template(folder_name, T, sampling_sweep)
    copy_common_files(folder_name)
    run_sbatch(folder_name)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python template.py <temperature> <sampling_sweep>")
        sys.exit(1)

    try:
        T_input = int(sys.argv[1])
    except ValueError:
        print("Error: Temperature must be an integer.")
        sys.exit(1)

    sampling_sweep_input = float(sys.argv[2])
    main(T_input, sampling_sweep_input)
