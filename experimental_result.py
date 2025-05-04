#standard libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

#toroidal filament functions
from process_probe_data import retreive_plasma_current, retreive_magnetic_signal,trim_quantities
from toroidalFilament_dir.plasma_shift import toroidal_filament_shift_progression
from toroidalFilament_dir.geometry_TT1 import all_arrays

#OFIT
from OFIT_dir.OFIT import OFIT
from OFIT_dir.local_image import rev_image, get_frames_for_shot

# shot_lst = list(range(961,968))
shot_lst = [370,433,582,665,920,926,969,1108,1275,1745,1804,2308]
time_extension = 40 #ms

for shot_no in shot_lst:
    #retreive processed data
    recorded_plasma_current, recorded_time, discharge_begin, discharge_end = retreive_plasma_current(shot_no)
    recorded_magnetic_signal = retreive_magnetic_signal(shot_no)

    end_time = min(discharge_begin + time_extension, discharge_end)

    time, plasma_current, plasma_signal = trim_quantities(recorded_time,recorded_magnetic_signal,recorded_plasma_current,discharge_begin,end_time)

    #calculate shift with toroidal filament
    use_probes = [[1,4,7,10],[12,3,6,9],[12,2,6,8]]
    valid_time, toroidal_R0_arr, toroidal_R0_err, toroidal_Z0_arr, toroidal_Z0_err = toroidal_filament_shift_progression(time,plasma_signal,use_probes)

    #calculate shift with OFIT
    all_frames = get_frames_for_shot(shot_no)
    all_frames_images = [rev_image(shot_no,frame) for frame in tqdm(all_frames)]

    frame_to_time = lambda frame: frame/2 + 260

    time_arr = []
    R0_arr, Z0_arr, r_arr = [], [], []
    R0_err_arr, Z0_err_arr, r_err_arr = [], [], []
    for frame_no, img in tqdm(enumerate(all_frames_images, start=1), total=len(all_frames_images), desc="Processing Frames"):
        #determine time
        OFIT_time = frame_to_time(frame_no)

        if OFIT_time < discharge_begin or end_time < OFIT_time:
            continue

        #calculate shift with OFIT
        result, cov = OFIT(img,shot_no,frame_no)

        if result == (None,None,None) or cov is None:
            continue

        R0,Z0,r = result

        time_arr.append(OFIT_time)

        R0_arr.append(R0-0.65) #subtract by major radius of tokamak to obtain shift value
        Z0_arr.append(Z0)
        r_arr.append(r)

        R0_err, Z0_err, r_err = cov.diagonal()
        R0_err_arr.append(R0_err)
        Z0_err_arr.append(Z0_err)
        r_err_arr.append(r_err)



    OFIT_time = np.array(time_arr)
    OFIT_Rshift, OFIT_Rerr = np.array(R0_arr), np.array(R0_err_arr)
    OFIT_Zshift, OFIT_Zerr= np.array(Z0_arr), np.array(Z0_err_arr)
    OFIT_r, OFIT_rerr = np.array(r_arr), np.array(r_err_arr)

    #plotting 

    def result_plot(axR, axZ, adjust):
        # toroidal filament result
        for t, R_shift, R_err, probe_arr in zip(valid_time, toroidal_R0_arr, toroidal_R0_err, use_probes):
            R_shift = np.array(R_shift)
            if adjust:
                if probe_arr == [12, 3, 6, 9]:
                    factor = 0.35
                elif probe_arr == [12, 2, 6, 8]:
                    factor = 0.23
                elif probe_arr == [1, 4, 7, 10]:
                    factor = 0.2
                elif probe_arr == [11,2, 5, 8]:
                    factor = 0.3
                else: factor = 0
            else: factor = 0
            line = axR.plot(t, R_shift - factor, label=f"{probe_arr}")
            color = line[0].get_color()
            axR.errorbar(t, R_shift - factor, yerr=R_err, alpha=0.1, color=color)

        # OFIT result
        axR.plot(OFIT_time, OFIT_Rshift, color="black", label="OFIT")
        axR.errorbar(OFIT_time, OFIT_Rshift, yerr=OFIT_Rerr, alpha=0.1, color="black")

        axR.set_xlim(discharge_begin, end_time)
        axR.set_ylim(-0.3, 0.3)
        axR.set_xlim(left=axR.get_xlim()[0], right=end_time)
        axR.grid()
        axR.set_xlabel("time [ms]")
        axR.set_ylabel("R shift [m]")
        axR.set_title("plasma horizontal shift")

        # toroidal filament result
        for t, Z_shift, Z_err, probe_arr in zip(valid_time, toroidal_Z0_arr, toroidal_Z0_err, use_probes):
            Z_shift = np.array(Z_shift)
            if adjust:
                if probe_arr == [12, 3, 6, 9]:
                    factor = 0.1
                elif probe_arr == [12, 2, 6, 8]:
                    factor = 0.05
                elif probe_arr == [1, 4, 7, 10]:
                    factor = 0.04
                elif probe_arr == [2, 5, 8, 11]:
                    factor = 0.01
                else: factor = 0
            else: factor = 0
            line = axZ.plot(t, Z_shift - factor, label=f"{probe_arr}")
            color = line[0].get_color()
            axZ.errorbar(t, Z_shift - factor, yerr=Z_err, alpha=0.1, color=color)

        # OFIT result
        axZ.plot(OFIT_time, OFIT_Zshift, color="black", label="OFIT")
        axZ.errorbar(OFIT_time, OFIT_Zshift, yerr=OFIT_Zerr, alpha=0.1, color="black")

        axZ.set_xlim(discharge_begin, end_time)
        axZ.set_ylim(-0.3, 0.3)
        axZ.set_xlim(left=axZ.get_xlim()[0], right=end_time)
        axZ.grid()
        axZ.set_xlabel("time [ms]")
        axZ.set_ylabel("Z shift [m]")
        axZ.set_title("plasma vertical shift")
        axZ.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4), ncol=3)


    fig, ax = plt.subplots(2,3,figsize = (20,10))

    fig.suptitle(f"result of shot {shot_no}")

    #plasma current
    column_names = plasma_signal.columns[1:]
    ax[0,0].plot(time, plasma_current,color = "black")
    ax[0,0].set_xlabel("time [ms]")
    ax[0,0].set_ylabel("$I_p$ [A]")
    ax[0,0].grid()
    ax[0,0].set_title("plot of $I_p$ agains time")

    cmap = plt.get_cmap("tab20")  # A colormap with 10 distinct colors

    #probe signal
    for i, name in enumerate(column_names):
        ax[1,0].plot(time, plasma_signal[name], label=name, color=cmap(i),alpha = 0.8)
    ax[1,0].set_xlabel("time [ms]")
    ax[1,0].set_ylabel("$S_i$ [T]")
    ax[1,0].grid()
    ax[1,0].set_title("magnetic probe signal")
    ax[1,0].legend(loc = "lower center", bbox_to_anchor = (0.5,-0.4),ncol = 4,framealpha = 0.3)

    #unadjusted result
    result_plot(ax[0,1],ax[1,1],False)
    #adjusted result
    result_plot(ax[0,2],ax[1,2],True)

    save_path = r"C:\Users\pitit\Documents\02_MUIC_programming\ICPY_441_Senior_project_in_physics\plasmaColumnPosition\resources\result" + f"\\{shot_no}"
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    plt.clf()