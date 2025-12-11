# TT1PlasmaColumnPosition

This repository contains the python implemented code for calculation of Toroidal Filament Model (with linear time complexity) and Optical Boundary Reconstruction (OFIT) in Thailand Tokamak-1 with the aim to calculate plasma column position within the tokamak and provide foundation for future real-time negative-feedback control. The details of theoretical background and implication are shown in "ANALYSIS OF PLASMA POSITION IN THAILAND TOKAMAK-1 USING TOROIDAL FILAMENT MODEL.pdf" to be signed and publised in MUIC library database. 

# Usage

## data preparation
Before applying the models to calculate the plasma position in TT-1, the data must be prepared in advanced

A directory named `data/shot_number` must exist where "shot_number" is always replaced by the actual experimental shot number.

In this directory the following files must exist:  
    1) "IP1.txt"  
    2) "IT1.txt"  
    3) "IOH1.txt"   
    4) "IV2.txt"  
    5) "GBP1T.txt", ... ,"GBP12T"  
    6) shot_number.avi  

if a subdirectory called `imgs` does not exist, all frames from the .avi video will be extracted into this newly created folder.

See `data/1641` and `data/1643` for the format of working input files. **Be careful on time step within .txt files**

## model execution
`main.py` is to be executed to run the models.

Inside `main.py`, a section denoted with "Parameter setup" can be found. Parameters of the calculation may be edited within this region

# Directory Structure

## General Structure
"main.py" is the main script which executes the toroidal filament model calibration plane transformation from experimental data.

"data" directory contains every experimental data required by main.py to function.

## Backend Calculation
"methods_script" directory contains the functions of "toroidal_filament" and "OFIT" which are used in "main.py" to perform calculations.

### The Toroidal Filament Model
`TFM.py`: The heart of the Toroidal Filament Model. It combines all the other files to perform toroidal filament model calculation.  
`parameters.py` specify all the parameters required in the calculation.  
`coefficent_nested_dict.pkl` contains all the taylor polynomial coefficients used in this model.

### OFIT
`OFIT.py` combines all the calculation functions of different python files used for Optical Boundary Reconstruction.  
`parameter.py`: contains all the parameters required for OFIT such as ROIs.  
`TT1_port_pixel.pkl` stores all the excluded pixels for edge detection in TT-1 tokamak as a set of (x,y) pixels.  

## Simulation files
Simulation of toroidal filament model and OFIT used for error and run time analysis can be found in `simulation/simulation_toroidal_filament.py` and `simulation/simulation_OFIT.py`.

# Possible improvements
## Toroidal Filament Model
Currently, the toroidal filament model relies purely on plasma current $I_p$. The codebase in `methods_script/TFM` can be altered to incoporate the effects from the plasma density profile: $$I_p = \int j \ dA$$.  
Also, magnetic field contribution from vessle's eddies induced by the plasma can also be investigated.

## Optical Boundary Reconstruction
Current implementation of the optical boundary reconstruction perform edge detection on Full-HD image (1920x1080) which greatly deterior the computation speed. The image can be reduced to lower resolution but the conversion factor between pixel and world coordinate must also be changed accordingly.  

Furthermore, the full implementation of optical boundary reconstruction incoporating the camera calibration parameters can be incoperated into the model once the parameters for the camera is known.
