#standard libraries
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import os

#toroidal filament functions
from methods_toroidal_filament.process_probe_data import retreive_plasma_current, retreive_magnetic_signal,trim_quantities
from methods_toroidal_filament.plasma_shift import toroidal_filament_shift_progression
from methods_toroidal_filament.parameters import all_arrays

#OFIT
from methods_OFIT.OFIT import OFIT
from methods_OFIT.local_image import rev_image, get_frames_for_shot
from methods_OFIT.parameters import TT1_major_radius


plt.style.use("seaborn-v0_8-dark-palette")

#defined experimental shot numbers to be used
shot_lst = [969]

#extended time from discharge begin. (For full discharge duration use np.inf)
time_extension = 40 #ms

#function to convert frame number to time with given formula
frame_to_time = lambda frame: frame/2 + 260

for shot_no in shot_lst:
    #calculate noise removed signal, time steps, discharge begin time, discharge end time from experimental data
    try:
        recorded_plasma_current, recorded_time, discharge_begin, discharge_end = retreive_plasma_current(shot_no)
    except ValueError:
        print(f"discharge time can't be determined for shot {shot_no}")
        continue

    ### toroidal filament model ###

    recorded_magnetic_signal = retreive_magnetic_signal(shot_no)

    end_time = min(discharge_begin + time_extension, discharge_end) 

    #trim the quantities to be within time discharge_begin to end_time
    time, plasma_current, plasma_signal = trim_quantities(recorded_time,recorded_magnetic_signal,recorded_plasma_current,discharge_begin,end_time)

    #calculate shift with toroidal filament
    use_probes = [[1,2,7,8],[1,3,7,9],[1,4,7,10],[2,3,8,9],[2,4,8,10],[3,4,9,10]] #specify magnetic probes to be used
    #result for toroidal filament model
    valid_time, toroidal_R0_arr, toroidal_R0_err, toroidal_Z0_arr, toroidal_Z0_err = toroidal_filament_shift_progression(time,plasma_signal,use_probes)

    ### OFIT ###

    all_frames = get_frames_for_shot(shot_no) #find all frames number of given experimental shot
    all_frames_images = [rev_image(shot_no,frame) for frame in all_frames] #retreive all RGB images of given shot

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

    #plotting 

    fig, (axR, axZ) = plt.subplots(1,2,figsize = (8,6))

    def toroidal_filament_plot(ax,arr,arr_err):
        for t, shift, err, probe_arr in zip(valid_time, arr, arr_err,use_probes):
            line = ax.plot(t,shift,label = f"{probe_arr}")
            color = line[0].get_color()
            ax.errorbar(t,shift,yerr=err,alpha = 0.1, color = color)

    toroidal_filament_plot(axR,toroidal_R0_arr,toroidal_R0_err)
    toroidal_filament_plot(axZ,toroidal_Z0_arr,toroidal_Z0_err)

    axR.plot(OFIT_time, OFIT_Rshift, color="black", label="OFIT")
    axR.errorbar(OFIT_time, OFIT_Rshift, yerr=OFIT_Rerr, alpha=0.1, color="black")

    axZ.plot(OFIT_time, OFIT_Zshift, color="black", label="OFIT")
    axZ.errorbar(OFIT_time, OFIT_Zshift, yerr=OFIT_Zerr, alpha=0.1, color="black")

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

    save_path = os.path.join("result_plot", str(shot_no))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()