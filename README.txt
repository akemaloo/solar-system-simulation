Solar System Simulation Project
===============================

Author: Akmal Daniel
Course: Computer Simulation

Overview
--------
This project is an object-oriented Python simulation of the Solar System in two dimensions.
It models the Sun, Mercury, Venus, Earth, Mars, and Jupiter as a gravitational many-body
system. The code supports three numerical integration methods:
- Beeman
- Euler-Cromer
- Direct Euler

This submission includes code for:
1. The default solar-system simulation with animation
2. Experiment 1: orbital periods
3. Experiment 1 supplementary timesteps study
4. Experiment 2: energy conservation and integration-method comparison
5. Experiment 4: planetary alignment analysis

Units
-----
The simulation uses:
- distance in AU
- mass in Earth masses
- time in years

Files Included
--------------
README.txt
    This file. It describes the contents of the submission and explains how to run the code.

data/parameters_solar.json
    Input file containing the Sun and planet parameters.

solar_sim/__init__.py
    Package exports for the main simulation components.

solar_sim/body.py
    Body class representing one gravitating object.

solar_sim/constants.py
    Physical constants, default timestep, default simulation length, and reference orbital periods.

solar_sim/integrators.py
    Time-integration methods: Beeman, Euler-Cromer, and Direct Euler.

solar_sim/io_utils.py
    Functions for loading bodies from JSON and saving CSV outputs.

solar_sim/simulation.py
    Main Simulation class and SimulationResults container.

solar_sim/analysis.py
    Analysis utilities for energy calculation, orbital-period detection, timestep studies, and alignment analysis.

solar_sim/plotting.py
    Plotting and animation utilities for orbit, energy, period-comparison, and alignment figures.

runs/run_default_simulation.py
    Runs the default Beeman simulation, saves the default plots/CSV output, and opens the orbit animation.

runs/run_experiment1_periods.py
    Runs Experiment 1 and compares simulated orbital periods with reference values.

runs/run_experiment1_timestep_study.py
    Runs the timestep-sensitivity study for Experiment 1.

runs/run_experiment2_energy_methods.py
    Runs Experiment 2 and compares energy conservation for Beeman, Euler-Cromer, and Direct Euler.

runs/run_experiment4_alignment.py
    Runs Experiment 4 and detects planetary alignments using the mean-angle criterion.

Requirements
------------
Python 3.11 or later is recommended.

Required third-party packages:
- numpy
- matplotlib

Install them with:
pip install numpy matplotlib

How to Run
----------
Run all commands from the project root folder, i.e. the folder containing:
- data
- runs
- solar_sim
- README.txt

The recommended command format is the module form shown below.

If matplotlib is using a non-interactive backend, the scripts still save their output files
but they do not open plot or animation windows.

Default Simulation
------------------
python -m runs.run_default_simulation

This:
- runs the default Beeman simulation
- saves default output files in outputs/
- prints the detected orbital periods
- opens a matplotlib animation window when an interactive backend is available

Files written by the default simulation:
- outputs/default_energy.csv
- outputs/default_energy.png
- outputs/default_energy_error.png
- outputs/default_orbits.png

Experiment 1: Orbital Periods
-----------------------------
python -m runs.run_experiment1_periods

This:
- runs the Beeman simulation
- detects orbital periods automatically
- compares them with reference values
- prints a comparison table
- saves the Experiment 1 orbital-period outputs in outputs/experiment1/

Files written by this run:
- outputs/experiment1/experiment1_periods.csv
- outputs/experiment1/experiment1_period_comparison.png
- outputs/experiment1/experiment1_orbits.png

Experiment 1 Supplementary Study: Timestep Sensitivity
------------------------------------------------------
python -m runs.run_experiment1_timestep_study

This:
- repeats Experiment 1 for multiple timesteps
- compares period accuracy across timestep choices
- saves the Experiment 1 timestep-study outputs in outputs/experiment1/

Files written by this run:
- outputs/experiment1/experiment1_timestep_study.csv
- outputs/experiment1/experiment1_timestep_study.png
- outputs/experiment1/experiment1_timestep_summary.txt

Experiment 2: Energy Conservation and Integration Methods
---------------------------------------------------------
python -m runs.run_experiment2_energy_methods

This:
- compares Beeman, Euler-Cromer, and Direct Euler
- saves total-energy and relative-drift plots
- saves a CSV summary of energy drift in outputs/experiment2/

Files written by this run:
- outputs/experiment2/experiment2_energy_summary.csv
- outputs/experiment2/experiment2_total_energy_comparison.png
- outputs/experiment2/experiment2_abs_relative_drift_comparison.png
- outputs/experiment2/experiment2_beeman_relative_drift_zoom.png
- outputs/experiment2/experiment2_beeman_energy_history_zoom.png

Note:
This run can take several minutes because it performs long simulations for all three methods.

Experiment 4: Planetary Alignment
---------------------------------
python -m runs.run_experiment4_alignment

This:
- runs the Beeman simulation for a long time window
- detects alignments of Mercury, Venus, Earth, Mars, and Jupiter
- uses the mean-angle alignment criterion
- checks the alignment criterion at every simulation timestep
- excludes the artificial t = 0 starting alignment from the reported results
- saves CSV summaries and alignment plots in outputs/experiment4/
- samples the max-deviation plot every 5 days for a manageable figure size
- also saves interval plots for selected nontrivial thresholds

Files written by this run:
- outputs/experiment4/experiment4_alignment_events_5deg.csv
- outputs/experiment4/experiment4_threshold_summary.csv
- outputs/experiment4/experiment4_max_deviation_5deg.png
- outputs/experiment4/experiment4_alignment_intervals_5deg.png
- outputs/experiment4/experiment4_alignment_intervals_10deg.png
- outputs/experiment4/experiment4_alignment_intervals_15deg.png
- outputs/experiment4/experiment4_threshold_counts.png

Note:
This run can take several minutes because it scans 2000 years at every timestep.

End of file
