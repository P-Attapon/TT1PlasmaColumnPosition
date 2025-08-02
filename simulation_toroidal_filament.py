from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

import numpy as np
import pandas as pd

from methods_script.toroidal_filament.DxDz import cal_newton_DxDz as cal_DxDz
from methods_script.toroidal_filament.plasma_shift import toroidal_filament_shift_progression
from methods_script.toroidal_filament.signal_strength import coil_signal
from methods_script.toroidal_filament.parameters import coil_angle_dict, R0, R, all_arrays
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

plt.show()