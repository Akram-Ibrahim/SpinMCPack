import os, shutil, io, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import write, read
from ase.build import sort, make_supercell

import SpinMC 



# Input values
##################
n_cells = 30
temperatures = np.arange(1, 810, 10)  
J = 70.76/1000; A = -0.61/1000 # in eV
max_spin_group = 4
angular_res = 0.2
cutoff_distance = 3.41444 + 0.1  # on-lattice V-V distance + skin distance 
sampling_step = 1e6; sample_size = 1e5; sampling_interval = 10
model_type = 'XY'                # Choose from {'Ising', 'XY', '3D'}
##################

# Read structure
##################
# Read primitive structure
prim_struc = read('/home/common/akram/QMC/mc_spin/POSCAR_V_T-VSe2-conv')
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
    v_values = np.arange(0, 1+res_v, res_v)

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

neighbor_array = []

for i, at in enumerate(super_struc):
    neighbor_indices = SpinMC.find_nearest_neighbors(super_struc, i, cutoff_distance)
    neighbor_array.append(neighbor_indices)    

neighbor_array = np.array(neighbor_array)

print('Successfully got the neighbors array ....')
    



# Run Monte Carlo
##################
# Initiate MC simulator
mc_simulator = SpinMC.Spin_MonteCarlo_Simulator(super_struc, temperatures, J, A, max_spin_group, orientations_lst, neighbor_array)
# Run
mc_simulator.run_simulation_range(sampling_step, sample_size, sampling_interval)
##################

print("Finished successfully ...")


