from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from .plasma_shift import toroidal_filament_shift_progression
from .signal_strength import coil_signal
from .geometry_TT1 import coil_angle_dict, R0, R
### simulate magnetic probe signal

np.random.seed(0)

def simulate_signal(num_iteration = 1_000):
    R_sim, Z_sim = [], []
    all_probe_signal = [[] for _ in range(13)]

    for _ in range(num_iteration):
        R_est = R_sim[-1] if len(R_sim) > 0 else 0
        Z_est = Z_sim[-1] if len(Z_sim) > 0 else 0
        if abs(R_est) <= 0.15:
            R_shift = R_est + np.random.choice([-0.001, 0.001], p = [0.8,0.2])
            Z_shift = Z_est + np.random.choice([-0.001, 0.001], p = [0.5,0.5])

        else:
            R_shift = R_est + np.random.choice([-0.001, 0.001], p = [0.5,0.5])
            Z_shift = Z_est + np.random.choice([-0.001, 0.001], p = [0.5,0.5])

        #append shift value
        R_sim.append(R_shift)
        Z_sim.append(Z_shift)

        all_probe_signal[0].append(0)

        for probe_num, probe_signal in enumerate(all_probe_signal[1:], start = 1):
            phi = coil_angle_dict[probe_num]
            r_probe = R0 + R * np.cos(phi)
            z_probe = R * np.sin(phi) - Z_shift
            a_f = R0 + R_shift
            signal_i = coil_signal(phi,r_probe,z_probe,a_f) #signal of probe at this specific iteration
            probe_signal.append(signal_i)

    return list(range(num_iteration)),R_sim, Z_sim, all_probe_signal

iteration, R_sim, Z_sim, probe_signal = simulate_signal()

signal_df = pd.DataFrame(np.array(probe_signal).T)
iteration_df = pd.Series(np.array(iteration).T)

# ### calculate plasma shift
use_probes = [[11, 12, 5, 6], [11, 1, 5, 7], [11, 2, 5, 8], [11, 3, 5, 9], [11, 4, 5, 10], [12, 1, 6, 7], [12, 2, 6, 8], [12, 3, 6, 9], [12, 4, 6, 10], [1, 2, 7, 8], [1, 3, 7, 9], [1, 4, 7, 10], [2, 3, 8, 9], [2, 4, 8, 10], [3, 4, 9, 10]]
valid_iteration, R_arr, R_err, Z_arr, Z_err =  toroidal_filament_shift_progression(iteration_df,signal_df,use_probes)

fig, ax = plt.subplots(1,2,figsize = (10,5))

ax[0].plot(iteration, R_sim, label = "R sim")
for iter, R,Re, probes in zip(valid_iteration, R_arr,R_err,use_probes):
    line, = ax[0].plot(iter,R,label = f"{probes}")
    ax[0].errorbar(iter,R,yerr = Re,color = line.get_color())
ax[0].set_xlabel("iteration [1]")
ax[0].set_ylabel("R shift [m]")
ax[0].grid()

ax[1].plot(iteration, Z_sim)
for iter, Z,Ze, probes in zip(valid_iteration, Z_arr,Z_err, use_probes):
    line, = ax[1].plot(iter,Z)
    ax[1].errorbar(iter,Z,yerr=Ze, color = line.get_color())
ax[1].set_xlabel("iteration [1]")
ax[1].set_ylabel("Z shift [m]")
ax[1].grid()

for a in ax:
    a.set_ylim(-0.2,0.2)

handles, labels = [], []
for axis in ax:
    h, l = axis.get_legend_handles_labels()
    handles += h
    labels += l

# Place the figure legend
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # or even 0.35 if needed
plt.show()


R_errors,R_sd = [],[]
Z_errors,Z_sd = [],[]

for R, Re, Z, Ze, probes in zip(R_arr, R_err, Z_arr, Z_err, use_probes):
    R_error = np.abs(np.array(R[1:]) - np.array(R_sim))  # exclude first element
    Z_error = np.abs(np.array(Z[1:]) - np.array(Z_sim))

    # Save for later
    R_errors.extend(R_error)
    Z_errors.extend(Z_error)

    R_sd.extend(Re)
    Z_sd.extend(Ze)

def mk_histogram(arr,name):
    save_path = r"C:\Users\pitit\Documents\01_MUIC_work\ICPY 441 Senior project\meetings\specialFilamentMeeting\\"
    # ====> Print overall mean error
    print(f"{name}: {np.mean(arr)}")

    # ====> Plot histogram
    plt.figure(figsize=(8,5))
    plt.hist(arr, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Error [m]")
    plt.ylabel("Count")
    plt.title(name)
    plt.grid(True)
    plt.savefig(save_path + name)


mk_histogram(R_errors,"high_overall_R_error")
mk_histogram(Z_errors,"high_overall_Z_error")
mk_histogram(R_sd,"high_overall_R_sd")
mk_histogram(Z_sd,"high_overall_Z_sd")