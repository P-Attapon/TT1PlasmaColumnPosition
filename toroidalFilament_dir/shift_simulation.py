from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from .plasma_shift import toroidal_filament_shift_progression
from .signal_strength import coil_signal
from .geometry_TT1 import coil_angle_dict, R0, R

### simulate magnetic probe signal

def simulate_signal(num_iteration = 1_000):
    R_sim, Z_sim = [], []
    all_probe_signal = [[] for _ in range(13)]

    for _ in range(num_iteration):
        R_est = R_sim[-1] if len(R_sim) > 0 else 0
        Z_est = Z_sim[-1] if len(Z_sim) > 0 else 0
        if abs(R_est) <= 0.12:
            R_shift = R_est + np.random.choice([-0.001, 0.001], p = [0.8,0.2])
            Z_shift = Z_est + np.random.choice([-0.001, 0.001], p = [0.2,0.8])

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

### calculate plasma shift
use_probes = [[1, 4, 7, 10], [12, 3, 6, 9], [2, 5, 8, 11], [12, 2, 6, 8]]
valid_iteration, R_arr, R_err, Z_arr, Z_err =  toroidal_filament_shift_progression(iteration_df,signal_df,use_probes)

fig, ax = plt.subplots(1,2,figsize = (15,5))

ax[0].plot(iteration, R_sim, label = "R sim")
for iter, R, probes in zip(valid_iteration, R_arr,use_probes):
    ax[0].plot(iter,R,label = f"{probes}")
ax[0].set_xlabel("iteration [1]")
ax[0].set_ylabel("R shift [m]")
ax[0].grid()
ax[0].legend()

ax[1].plot(iteration, Z_sim, label = "Z sim")
for iter, Z, probes in zip(valid_iteration, Z_arr, use_probes):
    ax[1].plot(iter,Z,label = f"{probes}")
ax[1].set_xlabel("iteration [1]")
ax[1].set_ylabel("Z shift [m]")
ax[1].grid()
ax[1].legend()

for a in ax:
    a.set_ylim(-0.2,0.2)

plt.show()