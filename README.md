# TT1PlasmaColumnPosition

This repository contains the python implemented code for calculation of Toroidal Filament Model and Optical Boundary Reconstruction (OFIT) in Thailand Tokamak-1 with the aim to calculate plasma column position within the tokamak and provide foundation for future real-time negative-feedback control. The details of theoretical background and implication are shown in "ANALYSIS OF PLASMA POSITION IN THAILAND TOKAMAK-1 USING TOROIDAL FILAMENT MODEL.pdf" to be signed and publised in MUIC library database. 

# Usage

## data preparation
Before applying the models to calculate the plasma position in TT-1, the data must be prepared in advanced

A directory named `data/shot_number` must exist where "shot_number" is always replaced by the actual experimental shot number.

In this directory the following files must exist:  
    1) "IP1.txt"  
    2) "IT1.txt"  
    3) "IOH1.txt"   
    4) "IV2.txt"  
    5) "GBPXT.txt" #Here "X" denote number 1 - 12  
    6) shot_number.avi  

if a subdirectory called `imgs` does not exist, all frames from the .avi video will be extracted into this newly created folder.

## model execution
`main.py` is to be executed to run the models.

Inside `main.py`, a section denoted with "Parameter setup" can be found. Parameters of the calculation may be edited within this region

`shot_lst` specifies number of all shots to run calculation on !the `data/shot_number` directory must exist before hand!

# Directory Structure
The structure of this directory is as followed:

"main.py" is the main script which executes the toroidal filament model calibration plane transformation from experimental data. In this file, it is possible to specify which models are to be used and plotted. 

"resources" directory contains every experimental data required by main.py to function.

"methods_script" directory contains the functions of "toroidal_filament" and "OFIT" which are used in "main.py" to perform calculations.

The "toroidal_filament" directory in "methods_script" contains different .py files used to perform the calculation. plasma_shift.py combines all the other files to perform toroidal filament model. parameters.py specify all the parameters used in the calculation such as major and minor radius and all set of magnetic probes defined for calculation. "coefficent_nested_dict.pkl" contains all the taylor polynomial coefficients used in this model. All the other .py files contain functions of calculation splitted into different subsections.

In "OFIT" subdirectory, "OFIT.py" combines all the calculation functions of different python files used for Optical Boundary Reconstruction. All the parameters required for OFIT such as ROIs are stored in "parameter.py". "TT1_port_pixel.pkl" stores all the excluded pixels for edge detection in TT-1 tokamak.

All the plots and images used in the publication are found in plotting.ipynb, and simulation of toroidal filament model and OFIT used for error and run time analysis can be found in "simulation_toroidal_filament.py" and "simulation_OFIT.py" respectively.
