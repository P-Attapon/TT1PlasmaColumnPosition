#standard libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

#toroidal filament functions
from process_probe_data import retreive_plasma_current, retreive_magnetic_signal,trim_quantities
from toroidalFilament_dir.plasma_shift import toroidal_filament_shift_progression
from toroidalFilament_dir.DxDz import cal_newton_DxDz as cal_DxDz
from toroidalFilament_dir.geometry_TT1 import coil_angle_dict, all_arrays

shot_lst = list(range(961,968))
# shot_lst = [966]
time_extension = 30 #ms

for shot_no in shot_lst:
    #retreive processed data
    recorded_plasma_current, recorded_time, discharge_begin, discharge_end = retreive_plasma_current(shot_no)
    recorded_magnetic_signal = retreive_magnetic_signal(shot_no)

    end_time = min(discharge_begin + time_extension, discharge_end)

    time, plasma_current, plasma_signal = trim_quantities(recorded_time,recorded_magnetic_signal,recorded_plasma_current,discharge_begin,end_time)

    #calculate shift with toroidal filament
    use_probes = [[1,4,7,10],[12,3,6,9],[11,2,5,8],[12,2,6,8]]
    # use_probes = [array for array in all_arrays if 11 not in array]
    valid_time, toroidal_R0_arr, toroidal_R0_err, toroidal_Z0_arr, toroidal_Z0_err = toroidal_filament_shift_progression(time,plasma_signal,use_probes)

    all_Dx = [[] for _ in range(len(use_probes))]
    all_Dz = [[] for _ in range(len(use_probes))]

    for signal in plasma_signal.to_numpy():
        for i,probe_num in enumerate(use_probes):
            Dx, Dz = cal_DxDz([signal[j] for j in probe_num],[coil_angle_dict[j] for j in probe_num])
            all_Dx[i].append(Dx)
            all_Dz[i].append(Dz)

    def result_plot(axR, axZ):
        # toroidal filament result
        for t, R_shift, R_err, probe_arr in zip(valid_time, toroidal_R0_arr, toroidal_R0_err, use_probes):
            line = axR.plot(t, np.array(R_shift), label=f"{probe_arr}")
            color = line[0].get_color()
            axR.errorbar(t, np.array(R_shift), yerr=R_err, alpha=0.1, color=color)

        axR.set_ylim(-0.5, 0.5)
        axR.set_xlim(left=discharge_begin, right=end_time)
        axR.grid()
        axR.set_xlabel("time [ms]")
        axR.set_ylabel("R shift [m]")
        axR.set_title("plasma horizontal shift")

        # toroidal filament result
        for t, Z_shift, Z_err, probe_arr in zip(valid_time, toroidal_Z0_arr, toroidal_Z0_err, use_probes):
            line = axZ.plot(t, np.array(Z_shift), label=f"{probe_arr}")
            color = line[0].get_color()
            axZ.errorbar(t, np.array(Z_shift), yerr=Z_err, alpha=0.1, color=color)

        axZ.set_ylim(-0.5, 0.5)
        axZ.set_xlim(left=discharge_begin, right=end_time)
        axZ.grid()
        axZ.set_xlabel("time [ms]")
        axZ.set_ylabel("Z shift [m]")
        axZ.set_title("plasma vertical shift")
        axZ.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4), ncol=3)

    fig, ax = plt.subplots(2,3,figsize = (15,8))

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
        ax[1,0].plot(time, -np.array(plasma_signal[name]), label=name, color=cmap(i),alpha = 0.8)
    ax[1,0].set_xlabel("time [ms]")
    ax[1,0].set_ylabel("$S_i$ [T]")
    ax[1,0].grid()
    ax[1,0].set_title("magnetic probe signal")
    ax[1,0].legend(loc = "lower center", bbox_to_anchor = (0.5,-0.4),ncol = 4,framealpha = 0.3)

    result_plot(ax[0,2],ax[1,2])

    for Dx_probe, Dz_probe,probe_num in zip(all_Dx, all_Dz,use_probes):
        ax[0,1].plot(time,Dx_probe,label = probe_num)
        ax[1,1].plot(time,Dz_probe,label = probe_num)


    ax[0,1].set_ylim(-0.4,0.4)
    ax[0,1].grid()
    ax[0,1].set_xlabel("time [ms]")
    ax[0,1].set_ylabel("Dx")
    ax[0,1].set_title(r"$\Delta_{||} \sim Dx$")

    ax[1,1].set_ylim(-0.4,0.4)
    ax[1,1].grid()
    ax[1,1].set_xlabel("time [ms]")
    ax[1,1].set_ylabel("Dz")
    ax[1,1].set_title(r"$\Delta_{\perp} \sim Dz$")
    ax[1,1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.4), ncol=3)

    save_path = r"C:\Users\pitit\Documents\02_MUIC_programming\ICPY_441_Senior_project_in_physics\plasmaColumnPosition\resources\result" + f"\\treeToroidal{shot_no}"
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)
