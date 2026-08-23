#standard libraries
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import os

#toroidal filament functions
from methods_script.toroidal_filament.process_probe_data import retreive_plasma_current
from methods_script.toroidal_filament.parameters import all_arrays, probe_lst_to_str, error_dict
from methods_script.toroidal_filament.TFM import TFM_main

# OFIT is imported LAZILY, inside the branch that uses it (see the dispatch
# block below). It pulls in opencv-python and scikit-learn, and importing it at
# module level made those hard requirements for every filament-only run -- which
# is not what this file needs. Do not move this import back to the top.

plt.style.use("seaborn-v0_8-dark-palette")

##################### Parameter setup ############################################
#`specifies number of all shots to run calculation on 
# !!!the `data/shot_number` directory must exist before hand!!! See README file
#shot_lst = [int(shot_num) for shot_num in os.listdir("data") if shot_num.isnumeric()]
shot_lst = [3970]

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
#
# ADAPTIVE: set use_probes = "adaptive" to let each sample pick the "best" set
# on its own. Rationale: one fixed set has a single reachable region in the
# linear (dU,dV) plane; where the plasma trajectory leaves it, Phi returns
# NaN/railed. Adaptive orders the sets by how much of their own reachable region
# is trustworthy (good-frac over the cached rt field, ties broken by median rt)
# and falls back to another set only on the samples the current one cannot
# represent, recovering domain-edge excursions.
#
# *** DEPENDENCY WARNING ***  When use_probes == "adaptive", the entire mprobe_*
# block below (weights, fit_ip, gains, weight_power, struct_ratio, rail_frac,
# min_samples, phys_step, uv_oversample) is IGNORED. Adaptive calls
# adaptive_select.adaptive_selection(), which uses its OWN internal weights
# source ("auto" here) and grid constants (adaptive_select.PHYS_STEP /
# UV_OVERSAMPLE = 0.0005 / 2.0). To change adaptive's behaviour, edit adaptive_select.py or the
# call in the dispatch block below -- NOT these variables. They apply only to the
# fixed-set path (use_probes = a list of probe lists).
use_probes = [[1,2,3,4,5,6,7,8,9,10,11,12]]

### ADDED: M-probe weighted least-squares method configuration ##########################
# use_mprobe = False -> original behaviour: use_probes must be 4-probe antipodal sets
#                       and displacement comes from cal_shift (the 2D map method).
# use_mprobe = True  -> use_probes may contain ANY number of probes (M >= 2), e.g.
#                       use_probes = [[1,2,3,4,5,6,7,8,9,10,11,12]]
#                       Displacement comes from the weighted linear estimator + its
#                       own 2D correction map (methods_script/.../mprobe.py).
# DEPENDENCY: use_probes = "adaptive" ALWAYS uses the M-probe estimator internally,
# regardless of this flag. Setting use_probes="adaptive" with use_mprobe=False is
# contradictory (the adaptive path still runs M-probe); keep use_mprobe=True when
# using adaptive to avoid confusion.
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

# NOTE: curation-gate thresholds (weight_power, struct_ratio, rail_frac,
# min_samples) and the Phi-map grid (phys_step, uv_oversample) are NOT exposed
# here. They have single, settled values that live at their source of truth:
#   - curation thresholds -> methods_script/toroidal_filament/curation.py
#       (WEIGHT_POWER=2.0, STRUCT_RATIO=6.0, RAIL_FRAC=0.01, MIN_SAMPLES=50)
#   - Phi grid            -> methods_script/toroidal_filament/mprobe.py
#       (PHYS_STEP=0.0005, UV_OVERSAMPLE=2.0)
# weight_power in particular is fixed at 2.0: it is the maximum-likelihood value
# for independent Gaussian errors and the only exponent for which the covariance
# output is a genuine position variance. Edit those modules to change a value in
# one place; that keeps main.py, compare_methods.py and adaptive_select consistent
# and avoids fragmenting the Phi cache (which is keyed on the grid).
#########################################################################################

# ---- config consistency guards (fail fast on contradictory settings) ----------
def _validate_config():
    # 1. adaptive requires the M-probe estimator; use_mprobe=False is contradictory.
    if use_probes == "adaptive" and not use_mprobe:
        raise ValueError(
            "use_probes='adaptive' requires use_mprobe=True (adaptive always uses "
            "the M-probe estimator internally). Set use_mprobe=True or use a fixed "
            "probe-set list instead of 'adaptive'.")
    # 2. use_probes must be either the string 'adaptive' or a list of probe lists.
    if use_probes != "adaptive":
        if not (isinstance(use_probes, (list, tuple)) and len(use_probes) > 0
                and all(isinstance(s, (list, tuple)) for s in use_probes)):
            raise ValueError(
                "use_probes must be the string 'adaptive' or a non-empty list of "
                f"probe-set lists, e.g. [[1,2,3,4]]. Got: {use_probes!r}")
        # 3. non-M-probe path only supports 4-probe antipodal sets.
        if not use_mprobe and any(len(s) != 4 for s in use_probes):
            raise ValueError(
                "use_mprobe=False supports only 4-probe antipodal sets, but "
                f"use_probes contains a set of a different length: {use_probes!r}. "
                "Set use_mprobe=True for arbitrary-length sets.")

_validate_config()
# -------------------------------------------------------------------------------


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

    if use_toroidal_filament_model and use_probes == "adaptive":
        # ADAPTIVE per-timestep set switching (see the use_probes comment above,
        # and adaptive_select.py). Per sample, keeps the current set while it is
        # still locally acceptable (inside its hull AND rt(u,v) <= RT_GOOD), else
        # walks a static, shot-independent order and takes the first set that
        # qualifies; inverts through that set's Phi -> REAL calibrated (dR, dZ).
        # If no set qualifies the sample is NaN, so the failure is visible in the
        # output rather than hidden in a summary statistic.
        # DEPENDENCY NOTES:
        #  - adaptive IGNORES mprobe_weights/_fit_ip/_gains/_phys_step/_uv_oversample
        #    set above; it uses adaptive_select's own weights_source + grid
        #    constants. To change adaptive weights, pass weights_source below;
        #    to change its grid, edit adaptive_select.PHYS_STEP.
        #  - weights_source="auto" uses THIS shot's pre-shot window, which is the
        #    right choice offline. Real time must pass "last" (inherited from a
        #    previous shot) because the pre-shot window is too short to compute
        #    weights live. Nothing else differs between the two.
        #  - This is the SAME function compare_methods.py calls: there is one
        #    selection scheme in the repo.
        from adaptive_select import adaptive_selection
        adaptive_result = adaptive_selection(shot_no, weights_source="auto")
        print(f"[adaptive] shot {shot_no}: coverage {adaptive_result['coverage']:.1%}, "
              f"{adaptive_result['n_switch']} switches, "
              f"order[0]={adaptive_result['order'][0]}, "
              f"provenance={adaptive_result['provenance']}")
        # adaptive_result carries t_ms, dR_m, dZ_m (real positions, metres). Build
        # a displacement_df so downstream plotting has the same handle it expects.
        displacement_df = pd.DataFrame({
            "Time (ms)": adaptive_result["t_ms"],
            "adaptive R": adaptive_result["dR_m"],
            "adaptive Z": adaptive_result["dZ_m"]})
    elif use_toroidal_filament_model:
        use_probes_str = [probe_lst_to_str(probe_set) for probe_set in use_probes]
        # ADDED: pass the M-probe configuration when enabled (None -> original path)
        mprobe_cfg = ({"weights": mprobe_weights, "fit_ip": mprobe_fit_ip,
                       "gains": mprobe_gains}
                      if use_mprobe else None)
        # Omitted keys (weight_power, struct_ratio, rail_frac, min_samples,
        # phys_step, uv_oversample) fall back to their source-of-truth defaults in
        # curation.py / mprobe.py -- see the config-block note above.
        displacement_df = TFM_main(shot_path = data_directory, use_probe_set = use_probes_str,
                                   mprobe = mprobe_cfg)

    ### calibration plane ###
    if use_calibration_plane_transformation:
        from methods_script.OFIT.OFIT import calibration_plane_shift
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
                # Load both raw Ip channels for comparison
                try:
                    import pandas as _pd2
                    _base = os.path.join(data_directory)
                    _ip1 = _pd2.read_csv(os.path.join(_base, "IP1.txt"),
                                         sep=r"\s+", skiprows=8, header=None,
                                         names=["t","v"])
                    _ip2 = _pd2.read_csv(os.path.join(_base, "IP2.txt"),
                                         sep=r"\s+", skiprows=8, header=None,
                                         names=["t","v"])
                    axI.plot(_ip1["t"], _ip1["v"]/1e3, 'k-',  lw=1.2, alpha=0.5, label="raw $I_p$ (IP1)")
                    axI.plot(_ip2["t"], _ip2["v"]/1e3, 'b-',  lw=1.2, alpha=0.5, label="raw $I_p$ (IP2)")
                except Exception:
                    pass
                axI.plot(t_ms, ip_meas, 'k-', lw=2, label="resolved $I_p$ (used)")

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
