from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

import numpy as np
import pandas as pd

from methods_toroidal_filament.DxDz import cal_newton_DxDz as cal_DxDz
from methods_toroidal_filament.plasma_shift import toroidal_filament_shift_progression
from methods_toroidal_filament.signal_strength import coil_signal
from methods_toroidal_filament.parameters import coil_angle_dict, R0, R, all_arrays
### simulate magnetic probe signal

plt.rcParams.update({
    "font.size":15
})
plt.style.use("seaborn-v0_8-dark-palette")

np.random.seed(0)

def simulate_signal(num_iteration = 1_000):
    R_sim, Z_sim = [], []
    all_probe_signal = [[] for _ in range(13)]

    for _ in range(num_iteration):
        R_est = R_sim[-1] if len(R_sim) > 0 else 0
        Z_est = Z_sim[-1] if len(Z_sim) > 0 else 0
        if abs(R_est) <= 0.15:
            R_shift = R_est + np.random.choice([-0.001, 0.001], p = [0.5,0.5])
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
use_probes = [[1,4,7,10]]
valid_iteration, R_arr, R_err, Z_arr, Z_err =  toroidal_filament_shift_progression(iteration_df,signal_df,use_probes)

all_Dx = [[] for _ in range(len(use_probes))]
all_Dz = [[] for _ in range(len(use_probes))]


for signal in signal_df.to_numpy():
    for i,probe_num in enumerate(use_probes):
        Dx, Dz = cal_DxDz([signal[j] for j in probe_num],[coil_angle_dict[j] for j in probe_num])
        all_Dx[i].append(Dx)
        all_Dz[i].append(Dz)

fig, ax = plt.subplots(2,2,figsize = (10,5))

for Dx_arr, probes in zip(all_Dx,use_probes):
    ax[0,0].plot(iteration,Dx_arr)
ax[0,0].set_xlabel("iteration [1]")
ax[0,0].set_ylabel("Dx [m]")
ax[0,0].grid()

for Dz_arr, probes in zip(all_Dz,use_probes):
    ax[0,1].plot(iteration,Dz_arr)
ax[0,1].set_xlabel("iteration [1]")
ax[0,1].set_ylabel("Dz [m]")
ax[0,1].grid()

ax[1,0].plot(iteration, R_sim, label = "R sim")
for iter, R,Re, probes in zip(valid_iteration, R_arr,R_err,use_probes):
    line, = ax[1,0].plot(iter,R,label = f"{probes}")
    ax[1,0].errorbar(iter,R,yerr = Re,color = line.get_color())
ax[1,0].set_xlabel("iteration [1]")
ax[1,0].set_ylabel("R shift [m]")
ax[1,0].grid()

ax[1,1].plot(iteration, Z_sim)
for iter, Z,Ze, probes in zip(valid_iteration, Z_arr,Z_err, use_probes):
    line, = ax[1,1].plot(iter,Z)
    ax[1,1].errorbar(iter,Z,yerr=Ze, color = line.get_color())
ax[1,1].set_xlabel("iteration [1]")
ax[1,1].set_ylabel("Z shift [m]")
ax[1,1].grid()

num_row, num_col = ax.shape
for i in range(num_row):
    for j in range(num_col):
        ax[i,j].set_ylim(-0.2,0.2)

ax[0,0].set_ylim(-0.3,0.3)

# handles, labels = [], []
# for axis in ax:
#     h, l = axis.get_legend_handles_labels()
#     handles += h
#     labels += l

# # Place the figure legend
# fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.3)  # or even 0.35 if needed
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

def mk_histogram(arr, title, x_label, ax):
    # ====> Print overall mean error
    print(f"{title}: {np.mean(arr)}")

    mean = np.mean(arr)

    # ====> Plot histogram
    ax.hist(arr, bins=30)
    ax.axvline(mean, color="red", label="mean", lw=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True)

    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.set_xlim(left=0)
    ax.legend(loc="upper right")

# Create subplots
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Fill subplots
mk_histogram(R_errors, r"$absolute \ residual \ of \ \Delta_R$", r"$y - \hat{y} \ [m]$", ax[0, 0])
mk_histogram(Z_errors, r"$absolute \ residual \ of \ \Delta_Z$", r"$y - \hat{y} \ [m]$", ax[0, 1])
mk_histogram(R_sd, r"uncertainty histogram of $\Delta_R$", r"$\sigma$ [m]", ax[1, 0])
mk_histogram(Z_sd, r"uncertainty histogram of $\Delta_Z$", r"$\sigma$ [m]", ax[1, 1])

# Adjust layout and save once
plt.tight_layout()
save_path = r"C:\\Users\\pitit\\Documents\\01_MUIC_work\\ICPY 441 Senior project\\columnPositionPaper\\latex\images\\"
# plt.savefig(save_path + "all_histograms.png")
plt.close()