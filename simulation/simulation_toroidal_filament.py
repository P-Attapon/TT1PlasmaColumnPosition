from matplotlib import pyplot as plt
import sys
import os
import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use("QtAgg")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from methods_script.toroidal_filament.DxDz import cal_newton_DxDz as cal_DxDz
from methods_script.toroidal_filament.plasma_shift import toroidal_filament_shift_progression
from methods_script.toroidal_filament.signal_strength import coil_signal
from methods_script.toroidal_filament.parameters import coil_angle_dict, R0, R, all_arrays, probe_lst_to_str, I
### simulate magnetic probe signal
plt.rcParams.update({
    "font.size":15
})
plt.style.use("seaborn-v0_8-dark-palette")

save_path = "/home/piti-archlinux/Projects/TT1/TT1-EFIT"

np.random.seed(0)
def simulate_signal_df(dR_all, dZ_all):
    """
    Simulate signal at each magnetic probe given plasma displacement

    :param dR_all: array of plasma displcaement along R direction
    :param dZ_all: array of plasma displcaement along Z direction
    :return: simulated magnetic signal at all probes in TT1 corresponding to given displacement 
    """

    def simulate_signal_at_probe(probe_num, dR, dZ):
        """Helper function to calculate magnetic signal at 
        specific probe given plasma displacement"""

        phi = coil_angle_dict[probe_num] #defined polar angle of probe on poloidal plane

        ## location of probe on poloidal plane with center plasma loop's center ##
        r_probe = R0 + R * np.cos(phi) 
        z_probe = R * np.sin(phi) - dZ
        ##############################

        a_f = R0 + dR # plasma filament's radius

        return coil_signal(phi,r_probe,z_probe,a_f)
    
    #create dataframe to assign values
    signal_df = pd.DataFrame(np.nan, index = range(0,len(dR_all)),columns=["Time step"] + [f"GBP{i}T" for i in range(1,13)])

    #calculate magnetic signasl for each probe at each given displacement
    for i, (dR_i, dZ_i) in enumerate(zip(dR_all, dZ_all)):
        signal_i = [i] + [simulate_signal_at_probe(num,dR_i,dZ_i) for num in range(1,13)]

        #assign values into signal_df
        signal_df.iloc[i] = signal_i

    return signal_df

def add_absolute_err(df:pd.DataFrame):
    for col in df.columns:
        if col == "defined": continue
        df[f"{col} abs error"] = np.abs(df[col] - df["defined"])
    return

if __name__ == "__main__":
    #define simulation parameters

    R_amp, Z_amp = 0.1, 0.0

    iteration_array = pd.Series(np.linspace(0, 440, 200) / 1000.0)

    phase = 2 * np.pi * iteration_array / iteration_array.max()

    dR_all = pd.Series( 
        R_amp * np.sin(
            2 * np.pi * (iteration_array)
            / (iteration_array.iloc[-1])
        ), name = "defined"
    )

    dZ_all = pd.Series(
        Z_amp * np.cos(phase),
        name="defined"
    )

    use_probes = all_arrays # set of magnetic probes to use in toroidal filament model

    #simulate magnetic field table
    signal_df = simulate_signal_df(dR_all, dZ_all)

    tfm_df = pd.concat(
        [dR_all.rename("R0") + R0, dZ_all.rename("Z0"), signal_df],
        axis=1
        )
    tfm_df["Time step"] = iteration_array
    tfm_df["Ip"] = I * np.ones_like(iteration_array)

    tfm_df.to_csv(os.path.join(save_path,"tfm_df.csv"), index=False)
    

    # ### calculate plasma shift
    # valid_iteration, R_arr, R_err, Z_arr, Z_err =  toroidal_filament_shift_progression(iteration_array,signal_df,use_probes)

    # #convert calculation result of each probe into data frame
    # dR_df = pd.DataFrame(data = np.array(R_arr).transpose(), columns = [probe_lst_to_str(probe_set) for probe_set in use_probes])
    # dZ_df = pd.DataFrame(data = np.array(Z_arr).transpose(), columns = [probe_lst_to_str(probe_set) for probe_set in use_probes])

    # #adjust index of dataframe
    # # Due to predefined shift_value at (0,0) in plasma_shift.py the row indices of defined and calculated are mismatched

    # def adjust_index(df:pd.DataFrame):
    #     for col in df.columns:
    #         df[col] = df[col].iloc[1:].reset_index(drop=True)
    #     return
    # adjust_index(dR_df)
    # adjust_index(dZ_df)

    # #add defined value of displacement
    # dR_df = pd.concat([dR_df, dR_all], axis = 1)
    # dZ_df = pd.concat([dZ_df, dZ_all], axis = 1)

    # add_absolute_err(dR_df)
    # add_absolute_err(dZ_df)
    # error_txt_path = os.path.join("simulation","TFM_error")
    # dR_df.filter(like = "error",axis = 1).describe().to_string(os.path.join(error_txt_path,"radialError.txt"))
    # dZ_df.filter(like = "error",axis = 1).describe().to_string(os.path.join(error_txt_path, "verticalError.txt"))

    # print(f"Simulation statistic saved to {error_txt_path}")

    # fig, ax = plt.subplots(1,2,figsize = (10,5))

    # ax[0].plot(iteration_array, dR_all, label = "R sim", lw = 5, alpha = 0.5)
    # for iter, R,Re, probes in zip(valid_iteration, R_arr,R_err,use_probes):
    #     line, = ax[0].plot(iter,R,label = f"{probes}")
    #     ax[0].errorbar(iter,R,yerr = Re,color = line.get_color())
    # ax[0].set_xlabel("iteration [1]")
    # ax[0].set_ylabel(r"$\Delta_Z$ [m]")
    # ax[0].grid()

    # ax[1].plot(iteration_array, dZ_all, lw = 5, alpha = 0.5)
    # for iter, Z,Ze, probes in zip(valid_iteration, Z_arr,Z_err, use_probes):
    #     line, = ax[1].plot(iter,Z)
    #     ax[1].errorbar(iter,Z,yerr=Ze, color = line.get_color())
    # ax[1].set_xlabel("iteration [1]")
    # ax[1].set_ylabel(r"$\Delta_Z$ [m]")
    # ax[1].grid()

    # for a in ax:
    #     a.set_ylim(-0.3,0.3)
    #     a.set_xlim(0,100)

    # fig.suptitle(f"Data size = {len(iteration_array)}")

    # save_path = os.path.join("result_plot","shift_simulation","TFM_simulation_error")

    # plt.savefig(save_path)
    # print(f"Simulation image saved to {save_path}")
