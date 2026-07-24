#standard libraries
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import pandas as pd
import os
import cv2

#toroidal filament functions
from methods_script.toroidal_filament.process_probe_data import retreive_plasma_current, retreive_magnetic_signal,trim_quantities, calibrate_signal_df, read_txt,mk_noise_df
from methods_script.toroidal_filament.plasma_shift import toroidal_filament_shift_progression
from methods_script.toroidal_filament.parameters import all_arrays, calibration_coeff, R0, probe_lst_to_str, error_dict
from methods_script.toroidal_filament.TFM import TFM_main

#OFIT
from methods_script.OFIT.OFIT import OFIT, process_image, field_edge_detection, calibration_plane_shift
from methods_script.OFIT.transformation import RANSAC_circle
from methods_script.OFIT.local_image import rev_image, get_frames_for_shot
from methods_script.OFIT.parameters import TT1_major_radius
from methods_script.OFIT.extract_frames import extract_frames_from_video

plt.style.use("seaborn-v0_8-dark-palette")

##################### Parameter setup ############################################
#`specifies number of all shots to run calculation on 
# !!!the `data/shot_number` directory must exist before hand!!! See README file
#shot_lst = [int(shot_num) for shot_num in os.listdir("data") if shot_num.isnumeric()]
shot_lst = [1641]

#define what methods to use
use_toroidal_filament_model = True              #if false the model will be skipped
use_calibration_plane_transformation = False     #if false the model will be skipped

#If true, then save edge detection image of each frame in result_plot/edge_detection/shot_number
edge_detection_image = False

#frames interval to skip in calibration plane (frame_step = 1) to use all frames
frame_step = 1

#save path of final plot
save_directory = os.path.join("result_plot","calculation_result")

#specify magnetic probes GBPXT to be used 
#print(all_arrays) #to see all possible combinations 
#use_probes = all_arrays # to use every existing probe combination
# default: one well-conditioned example set; with use_mprobe=True any length works,
# e.g. use_probes = [[1,2,3,4,5,6,7,8,9,10,11,12]]
use_probes = [[1,2,3,4,5,6,7,8,9,10,11,12]]

### ADDED: M-probe weighted least-squares method configuration ##########################
# use_mprobe = False -> original behaviour: use_probes must be 4-probe antipodal sets
#                       and displacement comes from cal_shift (the 2D map method).
# use_mprobe = True  -> use_probes may contain ANY number of probes (M >= 2), e.g.
#                       use_probes = [[1,2,3,4,5,6,7,8,9,10,11,12]]
#                       Displacement comes from the weighted linear estimator + its
#                       own 2D correction map (methods_script/.../mprobe.py).
use_mprobe = True

# per-probe weights (curation input; probes not listed default to 1.0; 0.0 excludes)
# example: mprobe_weights = {3: 0.0, 7: 0.5}  -> probe 3 off, probe 7 half-weight
#
# SET mprobe_weights = "auto" to compute weights from Layer-1 curation:
#   w_i = 1/sigma_i^2 where sigma_i = std of the detrended pre-plasma residual,
#   with probes failing a data-integrity gate (railed / dropout / non-stationary
#   / no pre-plasma window) dropped to weight 0. Computed once per shot.
mprobe_weights = "auto"

# plasma current handling:
#   False -> use the measured IP1 at each timestep (2 unknowns; recommended)
#   True  -> fit the current as a 3rd unknown (cross-check mode)
mprobe_fit_ip = True

# per-probe gain/polarity calibration factors g_p (measured = g_p * physical);
# signals are divided by g_p. Negative g_p corrects a polarity-flipped probe.
# None -> all 1.0. NOTE: absolute-field methods REQUIRE these to be calibrated
# (curation task); the values below, if any, are only as good as their source.
# example: mprobe_gains = {11: -1.21, 12: -0.43}
mprobe_gains = None

# curation tuning (only used when mprobe_weights = "auto")
# weight exponent: w_i = 1/sigma_i**power. LEAVE AT 2.0 unless there is strong
# evidence that the dominant probe error is non-random (systematic/correlated)
# rather than random noise. 2.0 is the maximum-likelihood value for independent
# Gaussian errors and the only exponent for which the covariance output is a
# genuine position variance. See methods_script/.../curation.py for details.
mprobe_weight_power = 2.0

# validity-gate thresholds (data-integrity only; sigma does the grading).
#   struct_ratio: drop a probe if the second half of its pre-plasma residual is
#                 this many times noisier than the first half. ONE-SIDED:
#                 growing noise is a fault, shrinking noise is a settling
#                 transient and is not penalised. On shot 1641 all 12 probes
#                 fall in 1.3-3.6, so 6.0 only catches egregious cases.
#   rail_frac   : drop if this fraction of pre-plasma samples sit pinned at the
#                 channel min/max (saturation / stuck channel).
#   min_samples : minimum usable pre-plasma samples for sigma to be meaningful.
mprobe_struct_ratio = 6.0
mprobe_rail_frac = 0.01
mprobe_min_samples = 50

# Phi-map grid resolution (only used when use_mprobe = True).
#   phys_step: physical (dR,dZ) sweep step in metres for building the Phi map.
#              This single value sets BOTH grids - the (dU,dV) lookup grid
#              resolution is derived from it so the two stay consistent.
#              Accuracy is flat vs this on real data (grid error << noise floor);
#              1 mm is the paper convention, 2 mm builds ~4x faster with
#              identical real-data accuracy. Smaller = finer but slower to build.
#   uv_oversample: nudges the lookup grid finer than the sweep count (>1) for
#              unevenly-sampled edge regions; leave at 1.0 to keep grids matched.
mprobe_phys_step = 0.001
mprobe_uv_oversample = 1.0
#########################################################################################

#extended time from discharge begin. (For full discharge duration use np.inf)
time_extension = np.inf #ms

#########################################################################################################

for shot_no in shot_lst:
    data_directory = os.path.join("data", str(shot_no))
    try:
        recorded_plasma_current, recorded_time, discharge_begin, discharge_end = retreive_plasma_current(shot_no)
        end_time = min(discharge_begin + time_extension, discharge_end) 
        print(f"Shot number: {shot_no}, peak plasma current: {np.round(np.max(recorded_plasma_current)/1000,2)} kA")
    except ValueError:
        print(f"discharge time can't be determined for shot {shot_no}")
        continue

    if use_toroidal_filament_model:
        use_probes_str = [probe_lst_to_str(probe_set) for probe_set in use_probes]
        # ADDED: pass the M-probe configuration when enabled (None -> original path)
        mprobe_cfg = ({"weights": mprobe_weights, "fit_ip": mprobe_fit_ip,
                       "gains": mprobe_gains,
                       "weight_power": mprobe_weight_power,
                       "struct_ratio": mprobe_struct_ratio,
                       "rail_frac": mprobe_rail_frac,
                       "min_samples": mprobe_min_samples,
                       "phys_step": mprobe_phys_step,
                       "uv_oversample": mprobe_uv_oversample}
                      if use_mprobe else None)
        displacement_df = TFM_main(shot_path = data_directory, use_probe_set = use_probes_str,
                                   mprobe = mprobe_cfg)

    ### calibration plane ###
    if use_calibration_plane_transformation:
        calibration_plane_df = calibration_plane_shift(
            data_directory = data_directory, shot_no = shot_no, frame_step = frame_step, discharge_begin = discharge_begin,
            discharge_end = end_time, edge_detection_image = edge_detection_image
        )

    ### Plotting ###
    # Define color cycle (matplotlib default)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def error_line_overlay(ax, t, displacement, probe_set, direction, color, step=200):
        """Overlay scalar error bars at downsampled points.

        error_dict holds the paper's per-set simulation error, keyed only by the
        15 antipodal 4-probe sets. For M-probe sets (any other length/combination)
        there is no tabulated value, so the band is simply skipped rather than
        crashing. (A genuine M-probe uncertainty is available as the estimator
        covariance, but it lives in proxy space and would need mapping through Phi
        to become an R/Z error bar - not attempted here.)
        """
        err_value = error_dict.get(probe_set + direction)
        if err_value is None:
            return
        t_err = t[::step]
        dis_err = displacement[::step]

        ax.errorbar(
            t_err,
            dis_err,
            yerr=err_value,
            fmt='none',
            color=color,
            alpha=0.7,
            capsize=3
        )


    # Plot only if at least one method is active
    if use_toroidal_filament_model or use_calibration_plane_transformation:

        fig, (axR, axZ) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        label_colors = {}

        # -------------------------------
        # TFM displacement curves (R, Z)
        # -------------------------------
        if use_toroidal_filament_model:

            time = displacement_df["Time (ms)"]

            for col in displacement_df.columns:
                if col in ("Time (ms)", "IP"):
                    continue
                if col.endswith("Ifit"):     # fitted-current columns handled separately
                    continue

                disp = displacement_df[col]
                label = col[:-2]        # e.g. "14710"

                # Assign color per probe group
                if label not in label_colors:
                    label_colors[label] = colors[len(label_colors) % len(colors)]
                color = label_colors[label]

                if col.endswith("R"):
                    axR.plot(time, disp, color=color)
                    error_line_overlay(axR, time, disp, label, "R", color)

                elif col.endswith("Z"):
                    axZ.plot(time, disp, color=color, label=f"probe {label}")
                    error_line_overlay(axZ, time, disp, label, "Z", color)


        # -------------------------------
        # Calibration-plane circle results
        # -------------------------------
        if use_calibration_plane_transformation and len(calibration_plane_df) > 0:
            axR.errorbar(
                calibration_plane_df["time"],
                calibration_plane_df["x0"],
                yerr=calibration_plane_df["x0 err"],
                fmt=".-",
                color="k",
                capsize=3
            )

            axZ.errorbar(
                calibration_plane_df["time"],
                calibration_plane_df["y0"],
                yerr=calibration_plane_df["y0 err"],
                fmt=".-",
                color="k",
                capsize=3
            )

            axZ.plot([], [], color="k", label="Edge detection")

            calibration_plane_df.to_csv(os.path.join(save_directory,f'{shot_no}CCD.csv'))
        # --------------------------------
        # Axis labels, limits, extras
        # --------------------------------
        axR.set_ylabel(r"$\Delta_R$ [m]")
        axZ.set_ylabel(r"$\Delta_Z$ [m]")
        axZ.set_xlabel("time [ms]")

        axZ.legend(ncol=2, fontsize="small", loc="lower center", frameon=False)

        labels = ["(a)", "(b)"]
        for i, axis in enumerate((axR, axZ)):
            axis.set_ylim(-0.2, 0.2)
            # axis.set_xlim(time.min(), time.max())
            axis.text(
                0.02, 0.95, labels[i],
                transform=axis.transAxes,
                fontsize=16,
                fontweight='bold',
                va='top', ha='left'
            )

        axR.axhline(0, ls="--", color="gray", alpha=0.3)
        axZ.axhline(0, ls="--", color="gray", alpha=0.3)

        fig.suptitle(f"Shot {shot_no}", y=0.94)

        plt.tight_layout()

        # Save
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, f"{shot_no}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Calculation result saved to: {save_path}")

        plt.close()

        # -------------------------------
        # Fitted plasma current (only present when mprobe fit_ip=True)
        # -------------------------------
        if use_toroidal_filament_model:
            ifit_cols = [c for c in displacement_df.columns if c.endswith("Ifit")]
            if ifit_cols:
                t_ms = displacement_df["Time (ms)"]
                ip_meas_A = displacement_df["IP (A)"].to_numpy()
                ip_meas = ip_meas_A / 1e3   # kA
                figI, axI = plt.subplots(figsize=(8, 4))
                axI.plot(t_ms, ip_meas, 'k-', lw=2, label="measured $I_p$ (IP1)")

                # relative RMS error between fitted I0 and measured Ip, normalised
                # by RMS(Ip). RMS(Ip) is always positive and dominated by the
                # high-current part of the discharge, so no current gate is needed
                # (near-zero startup/shutdown samples contribute negligibly).
                # Only finite fitted values are included. Per fitted-current column.
                err_lines = []
                for c in ifit_cols:
                    ifit_A = displacement_df[c].to_numpy()
                    axI.plot(t_ms, ifit_A / 1e3, '--', lw=1.4,
                             label=f"fitted $I_0$ [{c[:-5]}]")
                    m = np.isfinite(ifit_A)
                    if m.any():
                        rms_diff = np.sqrt(np.mean((ifit_A[m] - ip_meas_A[m]) ** 2))
                        rms_ip = np.sqrt(np.mean(ip_meas_A[m] ** 2))
                        rel = rms_diff / rms_ip if rms_ip > 0 else float("nan")
                        err_lines.append(f"[{c[:-5]}]  rel. RMS err = {rel*100:.1f}%")

                axI.set_xlabel("time [ms]"); axI.set_ylabel("current [kA]")
                axI.set_title(f"Shot {shot_no}: fitted $I_0$ vs measured $I_p$")
                leg = axI.legend(fontsize=8, loc="upper right")
                axI.grid(alpha=0.3)

                # annotation just below the legend box (relative RMS error,
                # normalised by RMS(Ip))
                if err_lines:
                    txt = "rel. RMS error (norm. RMS $I_p$):\n" + "\n".join(err_lines)
                    axI.text(0.985, 0.74, txt, transform=axI.transAxes,
                             fontsize=7.5, ha="right", va="top",
                             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

                figI.tight_layout()
                ip_path = os.path.join(save_directory, f"{shot_no}_Ifit.png")
                figI.savefig(ip_path, dpi=200, bbox_inches='tight')
                print(f"Fitted-current plot saved to: {ip_path}")
                for line in err_lines:
                    print(f"    I0 vs Ip {line}")
                plt.close(figI)
