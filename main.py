#standard libraries
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import pandas as pd
import os
import cv2
from pathlib import Path

#toroidal filament functions
from methods_script.toroidal_filament.process_probe_data import retreive_plasma_current, retreive_magnetic_signal,trim_quantities, calibrate_signal_df, read_txt,path_full_shot_directory,mk_noise_df
from methods_script.toroidal_filament.plasma_shift import toroidal_filament_shift_progression
from methods_script.toroidal_filament.parameters import all_arrays, calibration_coeff, R0, probe_lst_to_str

#OFIT
from methods_script.OFIT.OFIT import OFIT, process_image, field_edge_detection
from methods_script.OFIT.transformation import RANSAC_circle
from methods_script.OFIT.local_image import rev_image, get_frames_for_shot
from methods_script.OFIT.parameters import TT1_major_radius

plt.style.use("seaborn-v0_8-dark-palette")

#define what methods to use
use_toroidal_filament_model = True
use_OFIT = False
use_calibration_plane_transformation = False

calibrate_magnetic_signal = True
edge_detection_image = False

frame_step = 5

#save path of final plot
save_directory = os.path.join("result_plot","OFIT_result")


use_probes = [[1,4,7,10],[2,4,8,10]] #specify magnetic probes to be used (all_arrays for all combination)

#dictionary of erro bars calculated from simulation_toroidal_filament.py
error_dict = {
    probe_lst_to_str([1,4,7,10]) + "R": 2e-03, probe_lst_to_str([2,4,8,10]) + "R": 10e-3,
    probe_lst_to_str([1,4,7,10]) + "Z": 2e-03, probe_lst_to_str([2,4,8,10]) + "Z": 2e-3,
}

#defined experimental shot numbers to be used
shot_lst = [1641]

#extended time from discharge begin. (For full discharge duration use np.inf)
time_extension = np.inf #ms

#function to convert frame number to time with given formula
frame_to_time = lambda frame: frame/2 + 260

#function to transform pixel to calibration plane
pixel_to_calibration = lambda q,edge_pixel, pixel_plane_ratio=0.9: (q - edge_pixel)*pixel_plane_ratio/1000

for shot_no in shot_lst:
    try:
        recorded_plasma_current, recorded_time, discharge_begin, discharge_end = retreive_plasma_current(shot_no)
        end_time = min(discharge_begin + time_extension, discharge_end) 
        print(f"Shot number: {shot_no}, peak plasma current: {np.round(np.max(recorded_plasma_current)/1000,2)} kA")
    except ValueError:
        print(f"discharge time can't be determined for shot {shot_no}")
        continue

    if use_toroidal_filament_model:
        #calculate noise removed signal, time steps, discharge begin time, discharge end time from experimental data

        ### toroidal filament model ###

        magnetic_signal = retreive_magnetic_signal(shot_no)

        if calibrate_magnetic_signal:
            shot_directory = os.path.join(path_full_shot_directory,str(shot_no))

            toroidal_current = read_txt(os.path.join(shot_directory,"IT1.txt"),["Time (ms)", "It"])
            ohmic_current = read_txt(os.path.join(shot_directory,"IOH1.txt"), ["Time (ms)", "Ioh"])
            vertical_current = read_txt(os.path.join(shot_directory,"IV2.txt"), ["Time (ms)", "Iv"])

            magnetic_signal = calibrate_signal_df(magnetic_signal,toroidal_current["It"],ohmic_current["Ioh"],vertical_current["Iv"])

        #trim the quantities to be within time discharge_begin to end_time
        time, plasma_current, plasma_signal = trim_quantities(recorded_time,magnetic_signal,recorded_plasma_current,discharge_begin,end_time)

        #calculate shift with toroidal filament
        #result for toroidal filament model
        valid_time, toroidal_R0_arr, toroidal_R0_err, toroidal_Z0_arr, toroidal_Z0_err = toroidal_filament_shift_progression(time,plasma_signal,use_probes)

    ### retreive images for OFIT and calibration plane transformation ###

    if use_OFIT or use_calibration_plane_transformation:
        all_frames = get_frames_for_shot(shot_no) #find all frames number of given experimental shot
        all_frames_images = [rev_image(shot_no,frame) for frame in all_frames] #retreive all RGB images of given shot

    ### OFIT ###
    if use_OFIT:
        all_rows = []
        for frame_no, img in tqdm(enumerate(all_frames_images, start=1), total=len(all_frames_images), desc="OFIT"):
            if frame_no % frame_step != 0: continue

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
        for frame_no, img in tqdm(enumerate(all_frames_images,start = 1),total=len(all_frames_images), desc="calibration plane"):    
            if frame_no % frame_step != 0: continue

            ### perform calculation only within discharge time ###
            calibration_plane_time = frame_to_time(frame_no)
            if calibration_plane_time < discharge_begin:continue
            if calibration_plane_time > end_time: break
            ###

            img_brightness = np.mean(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
            if img_brightness < 70 or img_brightness > 130: 
                continue

            # clean image and turn to grayscale
            processed_image = process_image(img,apply_hsv_mask=True)

            #detect plasma edge
            (x_high,y_high), (x_low,y_low) = field_edge_detection(processed_image)

            ### save image of plasma edge detection ###
            if edge_detection_image:
                x_com, y_com = np.append(x_high,x_low), np.append(y_high,y_low)
                img_detection = img.copy()
                for x, y in zip(x_com, y_com):
                    img_detection[y-3:y+3,x-3:x+3] = [255,0,0]
                output_dir = Path(os.path.join("result_plot","edge_detection",str(shot_no)))
                output_dir.mkdir(parents = True, exist_ok=True)
                filename = os.path.join(output_dir, str(frame_no) + ".jpg")
                mpimg.imsave(filename,img_detection)
            ######

            #transform y = 0 to be at center of image
            y_high, y_low = y_high - 1080//2, y_low - 1080//2

            #convert to calibration_plane

            u_high, v_high = pixel_to_calibration(x_high,500), pixel_to_calibration(y_high,0)
            u_low, v_low = pixel_to_calibration(x_low,500), pixel_to_calibration(y_low,0)

            (u0,v0,radius),*_ = RANSAC_circle(np.append(u_high,u_low), np.append(v_high,v_low))

            calibration_plane_rows.append([calibration_plane_time,u0 - R0,v0,radius])

        calibration_plane_df = pd.DataFrame(
            calibration_plane_rows,
            columns=["time","x0","y0","r"]
        )

    #plotting 
    def toroidal_filament_plot(ax, arr, direction, step=100):
        """
        ax        : matplotlib axis
        arr       : list/array of calculated values per probe
        direction : string to select error from error_dict
        step      : plot error bars every 'step' points
        """
        for t, shift, probe_arr in zip(valid_time, arr, use_probes):
            # Plot main line
            line = ax.plot(t, shift, label=f"{probe_arr}")
            color = line[0].get_color()

            # Select points for error bars
            t_err = t[::step]
            shift_err = shift[::step]

            # Get the scalar error from the dict
            err_value = error_dict[probe_lst_to_str(probe_arr) + direction]

            # Plot error bars only at selected points
            ax.errorbar(
                t_err,
                shift_err,
                yerr=err_value,   # scalar is fine
                alpha=0.7,
                color=color,
                fmt='none',
                capsize=3
            )

    if True in [use_toroidal_filament_model, use_OFIT, use_calibration_plane_transformation]:

        fig, (axR, axZ) = plt.subplots(1,2,figsize = (8,6))

        if use_toroidal_filament_model:    
            toroidal_filament_plot(axR,toroidal_R0_arr,"R")
            toroidal_filament_plot(axZ,toroidal_Z0_arr,"Z")

        if use_OFIT:
            axR.plot(OFIT_time, OFIT_Rshift, color="black", label="OFIT")
            axR.errorbar(OFIT_time, OFIT_Rshift, yerr=OFIT_Rerr, alpha=0.1, color="black")
            axZ.plot(OFIT_time, OFIT_Zshift, color="black", label="OFIT")
            axZ.errorbar(OFIT_time, OFIT_Zshift, yerr=OFIT_Zerr, alpha=0.1, color="black")
        
        if use_calibration_plane_transformation:
            axR.plot(calibration_plane_df["time"], calibration_plane_df["x0"],".--")
            axZ.plot(calibration_plane_df["time"], calibration_plane_df["y0"],".--", label = "calibration")

        axR.set_ylabel(r"$\Delta_R$ [m]")
        axR.set_title("centroid horizontal displacement")

        axZ.set_ylabel(r"$\Delta_Z$ [m]")
        axZ.set_title("centroid vertical displacement")

        axZ.legend(ncol=2, title = "Magnetic probe numbers")

        for ax in (axR,axZ):
            ax.set_xlim(discharge_begin, end_time)
            ax.set_ylim(-0.3,0.3)
            ax.grid()
            ax.set_xlabel("time [ms]")

        fig.suptitle(f"result of shot {shot_no}")

        save_path = os.path.join(save_directory, str(shot_no))

        plt.tight_layout()
        plt.show()
        # plt.savefig(save_path)
        # plt.clf()
