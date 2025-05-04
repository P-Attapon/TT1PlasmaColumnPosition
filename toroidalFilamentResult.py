#standard libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

#toroidal filament functions
from process_probe_data import retreive_plasma_current, retreive_magnetic_signal,trim_quantities
from toroidalFilament_dir.plasma_shift import toroidal_filament_shift_progression

shot_lst = list(range(961,968))
time_extension = 30 #ms

for shot_no in shot_lst:
    #retreive processed data
    recorded_plasma_current, recorded_time, discharge_begin, discharge_end = retreive_plasma_current(shot_no)
    recorded_magnetic_signal = retreive_magnetic_signal(shot_no)

    end_time = min(discharge_begin + time_extension, discharge_end)

    time, plasma_current, plasma_signal = trim_quantities(recorded_time,recorded_magnetic_signal,recorded_plasma_current,discharge_begin,end_time)

    #calculate shift with toroidal filament
    use_probes = [[1,4,7,10],[12,3,6,9],[11,2,5,8],[12,2,6,8]]
    valid_time, toroidal_R0_arr, toroidal_R0_err, toroidal_Z0_arr, toroidal_Z0_err = toroidal_filament_shift_progression(time,plasma_signal,use_probes)


    def result_plot(axR, axZ, adjust):
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

    fig, ax = plt.subplots(1,2,figsize = (15,5))
    result_plot(ax[0],ax[1],False)
    save_path = r"C:\Users\pitit\Documents\02_MUIC_programming\ICPY_441_Senior_project_in_physics\plasmaColumnPosition\resources\result" + f"\\treeToroidal{shot_no}"
    plt.tight_layout()
    plt.savefig(save_path)
