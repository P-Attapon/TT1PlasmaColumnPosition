#standard libraries
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
import cv2

#toroidal filament functions
from methods_script.toroidal_filament.process_probe_data import retrieve_plasma_current, retrieve_magnetic_signal,trim_quantities, read_txt, path_full_shot_directory
from methods_script.toroidal_filament.plasma_shift import toroidal_filament_shift_progression
from methods_script.toroidal_filament.parameters import all_arrays

#OFIT
from methods_script.OFIT.OFIT import OFIT, process_image, field_edge_detection
from methods_script.OFIT.transformation import RANSAC_circle
from methods_script.OFIT.local_image import rev_image, get_frames_for_shot
from methods_script.OFIT.parameters import TT1_major_radius

plt.style.use("seaborn-v0_8-dark-palette")

#define what methods to use
use_toroidal_filament_model = True
signal_calibration = True
use_OFIT = False
use_calibration_plane_transformation = True

#specify probe sets to be used
use_probes = [[2,4,8,10]] #specify magnetic probes to be used (all_arrays for all combination)

# Is the data in excel workbook or fullShotData
is_excel = False 

#defined experimental shot numbers to be used
shot_lst = [1641, 1643, 2766]

#extended time from discharge begin. (For full discharge duration use np.inf)
time_extension = 40 #ms

#function to convert frame number to time with given formula
frame_to_time = lambda frame: frame/2 + 260

#function to transform pixel to calibration plane
pixel_to_calibration = lambda q, edge_pixel, pixel_plane_ratio=1/0.9: (q - edge_pixel)*pixel_plane_ratio/1000

for shot_no in shot_lst:
    if use_toroidal_filament_model:
        #calculate noise removed signal, time steps, discharge begin time, discharge end time from experimental data

        # try:
        recorded_plasma_current, recorded_time, discharge_begin, discharge_end = retrieve_plasma_current(shot_no,is_excel)
        # except ValueError:
            # print(f"discharge time can't be determined for shot {shot_no}")
            # continue

        ### toroidal filament model ###

        recorded_magnetic_signal = retrieve_magnetic_signal(shot_no,is_excel)

        end_time = min(discharge_begin + time_extension, discharge_end) 

        #trim the quantities to be within time discharge_begin to end_time
        time, plasma_current, plasma_signal = trim_quantities(recorded_time,recorded_magnetic_signal,recorded_plasma_current,discharge_begin,end_time)

        #calculate shift with toroidal filament
        shot_directory = os.path.join(path_full_shot_directory,str(shot_no))

        _, toroidal_current = read_txt(os.path.join(shot_directory,"IT1.txt"))
        _, ohmic_current = read_txt(os.path.join(shot_directory,"IOH1.txt"))
        _, vertical_current = read_txt(os.path.join(shot_directory,"IV1.txt"))

        #result for toroidal filament model
        valid_time, toroidal_R0_arr, toroidal_R0_err, toroidal_Z0_arr, toroidal_Z0_err = toroidal_filament_shift_progression(time,plasma_signal,toroidal_current,ohmic_current,vertical_current,
                                                                                                                             use_probes,calibration=signal_calibration)

    ### retreive images for OFIT and calibration plane transformation ###

    if use_OFIT or use_calibration_plane_transformation:
        all_frames = get_frames_for_shot(shot_no) #find all frames number of given experimental shot
        all_frames_images = [rev_image(shot_no,frame) for frame in all_frames] #retreive all RGB images of given shot

    ### OFIT ###
    if use_OFIT:
        all_rows = []
        for frame_no, img in tqdm(enumerate(all_frames_images, start=1), total=len(all_frames_images), desc="OFIT"):
            #determine time
            OFIT_time = frame_to_time(frame_no)

            if OFIT_time < discharge_begin:continue
            if OFIT_time > end_time: break

            #calculate shift with OFIT
            (R0,Z0,r), cov = OFIT(img,shot_no,frame_no)

            if None in (R0,Z0,r) or cov is None:
                continue

            R0_err, Z0_err, r_err = cov.diagonal()

            new_row = [OFIT_time, R0-TT1_major_radius, Z0, r, R0_err,Z0_err,r_err]
            all_rows.append(new_row)
        
        OFIT_result = pd.DataFrame(
            all_rows,
            columns=["OFIT_time", "OFIT_R", "OFIT_Z", "OFIT_r", "OFIT_R_err", "OFIT_Z_err", "OFIT_r_err"]
        )

        OFIT_time = OFIT_result["OFIT_time"]
        OFIT_Rshift, OFIT_Rerr = OFIT_result["OFIT_R"], OFIT_result["OFIT_R_err"]
        OFIT_Zshift, OFIT_Zerr=  OFIT_result["OFIT_Z"], OFIT_result["OFIT_Z_err"]
        OFIT_r, OFIT_rerr =  OFIT_result["OFIT_r"], OFIT_result["OFIT_r_err"]

    ### calibration plane ###

    if use_calibration_plane_transformation:
        calibration_plane_rows = []
        for frame_no, img in tqdm(enumerate(all_frames_images,start = 1),total=len(all_frames_images), desc="OFIT"):    
            calibration_plane_time = frame_to_time(frame_no)
            if calibration_plane_time < discharge_begin:continue
            if calibration_plane_time > end_time: break

            img_brightness = np.mean(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
            if img_brightness < 70 or img_brightness > 130: 
                continue

            processed_image = process_image(img)

            (x_high,y_high), (x_low,y_low) = field_edge_detection(processed_image)
            y_high, y_low = y_high - 1080//2, y_low - 1080//2

            u_high, v_high = pixel_to_calibration(x_high,500), pixel_to_calibration(y_high,0)
            u_low, v_low = pixel_to_calibration(x_low,500), pixel_to_calibration(y_low,0)

            u_com, v_com = np.append(u_high,u_low), np.append(v_high,v_low)

            (x0,y0,radius),*_ = RANSAC_circle(np.append(u_high,u_low), np.append(v_high,v_low))

            calibration_plane_rows.append([calibration_plane_time,x0 - TT1_major_radius,y0,radius])

        calibration_plane_df = pd.DataFrame(
            calibration_plane_rows,
            columns=["time","x0","y0","r"]
        )

    #plotting 

    fig, (axR, axZ) = plt.subplots(1,2,figsize = (8,6))

    def toroidal_filament_plot(ax,arr,arr_err):
        for t, shift, err, probe_arr in zip(valid_time, arr, arr_err,use_probes):
            line = ax.plot(t,shift,label = f"{probe_arr}")
            color = line[0].get_color()
            ax.errorbar(t,shift,yerr=err,alpha = 0.1, color = color)

    if use_toroidal_filament_model:    
        toroidal_filament_plot(axR,toroidal_R0_arr,toroidal_R0_err)
        toroidal_filament_plot(axZ,toroidal_Z0_arr,toroidal_Z0_err)

    if use_OFIT:
        axR.plot(OFIT_time, OFIT_Rshift, color="black", label="OFIT")
        axR.errorbar(OFIT_time, OFIT_Rshift, yerr=OFIT_Rerr, alpha=0.1, color="black")
        axZ.plot(OFIT_time, OFIT_Zshift, color="black", label="OFIT")
        axZ.errorbar(OFIT_time, OFIT_Zshift, yerr=OFIT_Zerr, alpha=0.1, color="black")
    
    if use_calibration_plane_transformation:
        axR.plot(calibration_plane_df["time"], calibration_plane_df["x0"],".--")
        axZ.plot(calibration_plane_df["time"], calibration_plane_df["y0"],".--", label = "calibration")

    axR.set_ylabel(r"$\Delta_R$ [m]")
    axR.set_title("plasma horizontal shift")

    axZ.set_ylabel(r"$\Delta_Z$ [m]")
    axZ.set_title("plasma vertical shift")

    axZ.legend(ncol=2, title = "Magnetic probe numbers")

    for ax in (axR,axZ):
        ax.set_xlim(discharge_begin, end_time)
        ax.set_ylim(-0.3,0.3)
        ax.grid()
        ax.set_xlabel("time [ms]")

    fig.suptitle(f"result of shot {shot_no}")

    save_path = os.path.join("result_plot", "calibration_plane_result", f"{shot_no}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()