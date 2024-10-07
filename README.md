# SpinMC - Spin Monte Carlo Simulation Package

SpinMC is a Python package designed for performing Monte Carlo (MC) simulations of spin systems. It supports Ising, XY, and 3D Heisenberg models.

## Features
- Supports Ising, XY, and 3D Heisenberg models.
- Configurable parameters include temperature, interaction strengths, spin orientations, number of cells, magnetic interaction parameters, sample size, sampling interval, and angular resolution of spin orientations.
- Provides plots for energy and magnetization convergence.


## Prerequisites
- Python 3
- Required libraries: numpy, ase, pandas, matplotlib, tqdm, scipy

## Installation
Clone the repository:

## Usage
Edit `template.py`:
- Modify parameters such as number of cells, magnetic interaction parameters, sample size, sampling interval, angular resolution of spin orientations, etc., as needed.
- Usage: `python template.py <temperature> <sampling_sweep>`

This command runs the simulation using a submission script (`submission-script`) which can be adjusted according to user-specific computational resources.

## Output Files
- Energies: saved in the `energies` directory.
- Spin configurations: saved in the `spin_configs` directory.
- Convergence plots: saved as `Energy_<temperature>.png` and `Magnetization_<temperature>.png`.

## Key Components
- `Spin_MonteCarlo_Simulator`: Main class for running MC simulations.
- `monte_carlo_simulation()`: Executes the MC simulation.
- `calculate_site_energies()`: Calculates energies from neighboring spins.

## File Descriptions
- `template.py`: Script to initiate and run simulations.
- `SpinMC.py`: Core implementation of the `Spin_MonteCarlo_Simulator` class.


The model provided in this repository was utilized to generate the results discussed in the manuscript: [arXiv:2409.19082](https://arxiv.org/abs/2409.19082).
