# SpinMCPack - Spin Monte Carlo Simulation Package

SpinMCPack is a Python package designed for performing Monte Carlo (MC) simulations of spin systems. It supports Ising, XY, and 3D Heisenberg models. Is also features options for incorporating external magnetic and electric fields.

## Features
- Supports Ising, XY, and 3D Heisenberg models.
- Configurable parameters include temperature, spin orientations, supercell size, interaction parameters, number of equilibration steps, ensemble size, sampling interval, angular resolution of spins, and external field strength.
- Provides convergence and phase transition plots for energy, magnetization, topological charge, and their susceptibilities.


## Prerequisites
- Python 3
- Required libraries: numpy, ase, pandas, matplotlib, tqdm, scipy

## Installation
Clone the repository:

## Usage
Edit `template.py`:
- Modify parameters such as number of cells, magnetic interaction parameters, sample size, sampling interval, angular resolution of spin orientations, etc., as needed.
- Usage: `python template.py <temperature> <sampling_sweep> <B_z> <E_z>`

This command runs the simulation using a submission script (`submission-script`) which can be adjusted according to user-specific computational resources.

## Output Files
- Energies: saved in the `energies` directory.
- Spin configurations: saved in the `spin_configs` directory.
- Convergence plots are saved for Energy, Normalized Magnetization, and Topological Charge.

## Key Components
- `Spin_MonteCarlo_Simulator`: Main class for running MC simulations.
- `monte_carlo_simulation()`: Executes the MC simulation.
- `calculate_site_energies()`: Calculates energies from neighboring spins.

## File Descriptions
- `template.py`: Script to initiate and run simulations.
- `SpinMC.py`: Core implementation of the `Spin_MonteCarlo_Simulator` class.


The model provided in this repository was utilized to generate the results discussed in the manuscript: [arXiv:2409.19082](https://arxiv.org/abs/2409.19082).

![image](https://github.com/user-attachments/assets/db90d7af-3153-4b20-b046-08bb6c39e82a)

