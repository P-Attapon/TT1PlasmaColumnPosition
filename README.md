# TT1PlasmaColumnPosition

This repository contains the python implemented code for calculation of Toroidal Filament Model and Optical Boundary Reconstruction (OFIT) in Thailand Tokamak-1. "experimental_result.py" is the main file used to calculate result from experimental data in "resources" directory. Functions for toroidal filament model are stored in methods_toroidal_filament and methods_OFIT for functions of Optical Boundary Reconstruction. 

The "methods_toroidal_filament" directory contains different .py files used to perform the calculation. plasma_shift.py combines all the other files to perform toroidal filament model. parameters.py specify all the parameters used in the calculation such as major and minor radius and all set of magnetic probes defined for calculation. "coefficent_nested_dict.pkl" contains all the taylor polynomial coefficients used in this model. All the other .py files contain functions of calculation splitted into different subsections.

In methods_OFIT directory, "OFIT.py" combines all the calculation functions of different python files used for Optical Boundary Reconstruction. All the parameters required for OFIT such as ROIs are stored in "parameter.py". "TT1_port_pixel.pkl" stores all the excluded pixels for edge detection in TT-1 tokamak.

All the plots and images used in the publication are found in plotting.ipynb, and simulation of toroidal filament model and OFIT used for error and run time analysis can be found in "simulation_toroidal_filament.py" and "simulation_OFIT.py" respectively.
