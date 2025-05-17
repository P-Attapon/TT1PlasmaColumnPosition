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

plt.rcParams.update({
    "font.size":15
})
plt.style.use("seaborn-v0_8-dark-palette")

# shot_lst = [1108,1275,1745,1804,2308]
shot_lst = [2308]
time_extension = 30 #ms

for shot_no in shot_lst:
    #retreive processed data
    recorded_plasma_current, recorded_time, discharge_begin, discharge_end = retreive_plasma_current(shot_no)
    recorded_magnetic_signal = retreive_magnetic_signal(shot_no)

    end_time = min(discharge_begin + time_extension, discharge_end)

    time, plasma_current, plasma_signal = trim_quantities(recorded_time,recorded_magnetic_signal,recorded_plasma_current,discharge_begin,end_time)

    #calculate shift with toroidal filament
    use_probes = [arr for arr in all_arrays if 11 not in arr and 12 not in arr]
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
        plt.rcParams.update({
            "font.size":15
        })
        plt.style.use("seaborn-v0_8-dark-palette")
        # toroidal filament result
        for t, R_shift, R_err, probe_arr in zip(valid_time, toroidal_R0_arr, toroidal_R0_err, use_probes):
            line = axR.plot(t, np.array(R_shift))
            color = line[0].get_color()
            axR.errorbar(t, np.array(R_shift), yerr=R_err, alpha=0.1, color=color)

        axR.set_ylim(-0.3, 0.3)
        axR.set_xlim(left=discharge_begin, right=end_time)
        axR.grid()
        axR.set_xlabel("time [ms]")
        axR.set_ylabel(r"$\Delta_R$ [m]")

        # toroidal filament result
        for t, Z_shift, Z_err, probe_arr in zip(valid_time, toroidal_Z0_arr, toroidal_Z0_err, use_probes):
            line = axZ.plot(t, np.array(Z_shift), label=f"{probe_arr}")
            color = line[0].get_color()
            axZ.errorbar(t, np.array(Z_shift), yerr=Z_err, alpha=0.1, color=color)

        axZ.set_ylim(-0.3, 0.3)
        axZ.set_xlim(left=discharge_begin, right=end_time)
        axZ.grid()
        axZ.set_xlabel("time [ms]")
        axZ.set_ylabel(r"$\Delta_Z$ [m]")

        axZ.legend(loc="best", fontsize="small")

    fig, ax = plt.subplots(1,2,figsize = (8,6))

    fig.suptitle(f"toroidal filament model on shot {shot_no}")

    result_plot(ax[0],ax[1])

    save_path = r"C:\\Users\\pitit\\Documents\\01_MUIC_work\\ICPY 441 Senior project\\columnPositionPaper\\latex\\images" + f"\\treeToroidal{shot_no}"

    fig.subplots_adjust(bottom=0.25)

# Place global legend just below the subplots
    plt.tight_layout()
    plt.savefig(save_path)
