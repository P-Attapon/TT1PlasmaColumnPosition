import numpy as np
import pandas as pd
import os
from contextlib import ExitStack
from tqdm import tqdm
from matplotlib import pyplot as plt

from .parameters import all_arrays, shift_domain
from .plasma_shift import cal_shift
from .DxDz import cal_newton_DxDz as cal_dXdZ
from .process_probe_data import magnetic_field_calibration, calibration_coeff

taylor_order = 3 #order of taylor series fitting => must match with fitting coefficient file!

def determine_unique_probes(probe_set:list[str]) -> np.ndarray[str]:
    """determine all unique probe numbers in probe_set"""
    unique_probes = np.unique(" ".join(probe_set).split(" "))
    return unique_probes

def check_missing_files(shot_path:str,required_files: list[str]) -> None:
    """
    raise FileNotFoundError if shot_path does not exist or missing required txt files with
    """

    #check if directory exist
    if not os.path.isdir(shot_path):
        raise FileNotFoundError(f"directory {shot_path} does not exist")

    #list out all existing files in shot_path
    existing_files = set(os.listdir(shot_path))

    #determine missing files
    missing = required_files - existing_files

    #if missing is not empty raise Error
    if bool(missing):
        # sort missing by characters for easy reading
        missing_lst = list(missing)
        missing_lst.sort()
        raise FileNotFoundError(f"{missing_lst} are missing from {shot_path}")

    return

def correct_magnetic_signal(signal_dict: dict, It_val: float, Ioh_val: float, Iv_val: float) -> dict:
    """Remove coil-field pickup from probe signals.

    It_val, Ioh_val, Iv_val: resolved current values for this timestep,
    supplied by the caller (from current_channels.resolve_all output) rather
    than read from signal_dict directly. This decouples pickup subtraction from
    hardcoded channel names.
    """
    corrected_signal = {key: None for key in signal_dict.keys() if "GBP" in key}

    for key in corrected_signal.keys():
        probe_num = key[3:5] if key[3:5].isnumeric() else key[3]
        kt  = calibration_coeff["k" + probe_num + "t"]
        koh = calibration_coeff["k" + probe_num + "oh"]
        kv  = calibration_coeff["k" + probe_num + "v"]
        corrected_signal[key] = magnetic_field_calibration(
            signal_dict[key], kt, It_val, koh, Ioh_val, kv, Iv_val
        )

    corrected_signal["Time (ms)"] = signal_dict["Time (ms)"]
    corrected_signal["IP"] = signal_dict["IP"]   # resolved Ip value for this step
    return corrected_signal

def restrict_displacement(displacement_val:float, shift_domain:float) -> float:
    if abs(displacement_val) > shift_domain:
        if displacement_val < -shift_domain: displacement_val = -shift_domain
        elif displacement_val > shift_domain: displacement_val = shift_domain
    return displacement_val


def TFM_main(shot_path: str,use_probe_set: list[str],discharge_current:float=2500, discharge_offset: float = 100, mprobe: dict = None) -> pd.DataFrame:
    """
    Calculate plasma column position displacement with the Toroidal Filament Model
    :param shot_path: path to data directory containing
                      "IP1.txt", "IP2.txt", "IT1.txt", "IT2.txt",
                      "IOH1.txt", "IOH2.txt", "IV1.txt", "IV2.txt", and
                      "GBPXT.txt" for all X as unique probe numbers in probe_set.
                      *2 files are optional (tolerated if absent); they are used by
                      the current_channels resolver for redundant-channel health
                      checking and averaging (IT, IOH, IV) or fallback (IV).
    :param use_probe_set: set of magnetic probes to use for calculation
                      (set number must exist in all_arrays)
    :param discharge_current: threshold for begin and end of discharge
    :param discharge_offset: constant offset value helps to determine ending of discharge
    :param mprobe: ADDED (M-probe generalization). None -> original behaviour
                   (4-probe antipodal sets via cal_shift / the 2D map).
                   Otherwise a dict enabling the M-probe weighted least-squares
                   estimator (methods_script/toroidal_filament/mprobe.py) for
                   probe sets of ANY length M >= 2:
                     {"weights": None or {probe_number: weight, ...},
                      "fit_ip": False}   # True fits Ip as a 3rd unknown
                   Probes missing from the weights dict get weight 1.0.
    :return: dataframes of centroid displacement calculated from all probe_set
             along radial and vertical directions
    """
    number_of_probe_set = len(use_probe_set)
    #determine all unique magnetic probes to use
    unique_probes = determine_unique_probes(use_probe_set)

    # ADDED (M-probe): build one estimator per probe set, once per shot.
    # Weights and Phi map are fixed for the shot; the condition number of the
    # normal matrix is printed as a per-set health check.
    mprobe_est = None
    if mprobe is not None:
        from .mprobe import MProbeEstimator
        raw_w = mprobe.get("weights")
        gdict = mprobe.get("gains") or {}          # ADDED: per-probe gain factors
        fit_ip = bool(mprobe.get("fit_ip", False))
        # ADDED (curation): weights source is one of
        #   "auto" -> compute w_i = 1/sigma_i^2 from the pre-plasma residual
        #             (Layer-1 curation) once per shot, and persist to the disk
        #             cache so all sets on this shot reuse it and "last" can find it.
        #   "last" -> load the most recently persisted weights (real-time use:
        #             the current preshot window is NOT read; the previous shot's
        #             weights are reused, giving a fixed vector -> precomputable
        #             Phi maps). See weights_cache.py.
        #   dict   -> explicit per-probe weights, used verbatim.
        #   None   -> unit weights (all 1.0).
        auto_curation = (raw_w == "auto")
        use_last = (raw_w == "last")
        wdict = {} if (raw_w is None or auto_curation or use_last) else raw_w
        if auto_curation:
            from .curation import compute_weights
            from . import weights_cache
            all_probes = sorted({int(p) for s in use_probe_set for p in s.split()})
            wdict, sig_dbg, valid_dbg = compute_weights(
                shot_path, all_probes, discharge_current=discharge_current,
                power=mprobe.get("weight_power"),
                struct_ratio=mprobe.get("struct_ratio"),
                rail_frac=mprobe.get("rail_frac"),
                min_samples=mprobe.get("min_samples"))
            weights_cache.save_weights(shot_path, wdict)   # persist for reuse / "last"
            print("[curation] w_i = 1/sigma_i^%s from pre-plasma residual:"
                  % (mprobe.get("weight_power") or "2.0"))
            for p in all_probes:
                flag = "" if valid_dbg[p] else "  <- GATED (dropped)"
                print(f"    GBP{p:<2d} sigma={sig_dbg[p]:.3e} T  w={wdict[p]:.3e}{flag}")
        elif use_last:
            from . import weights_cache
            loaded = weights_cache.load_latest()
            if loaded is None:
                print("[curation] weights='last' but no stored weights found; "
                      "falling back to unit weights. Run an 'auto' shot first.")
                wdict = {}
            else:
                wdict, src_shot = loaded
                print(f"[curation] weights='last': reusing stored weights from "
                      f"shot {src_shot} (real-time mode; current preshot NOT read).")
        mprobe_est = {}
        for set_str in use_probe_set:
            probes = list(map(int, set_str.split()))
            weights = [float(wdict.get(p, 1.0)) for p in probes]
            gains = [float(gdict.get(p, 1.0)) for p in probes]
            est = MProbeEstimator(probes, weights=weights, fit_ip=fit_ip, gains=gains,
                                  phys_step=mprobe.get("phys_step"),
                                  uv_oversample=mprobe.get("uv_oversample"))
            mprobe_est[set_str] = est
            print(f"[mprobe] set [{set_str}] M={len(probes)} fit_ip={fit_ip} "
                  f"condition number={est.cond:.3g}")

    # Required input files. The *2 channel variants are OPTIONAL: the
    # current_channels resolver uses them for redundancy/health checking when
    # present, and falls back to the primary channel when absent.
    required_core = set(
        ["IP1.txt", "IT1.txt", "IOH1.txt", "IV2.txt"] +
        ["GBP" + i + "T.txt" for i in unique_probes]
    )
    check_missing_files(shot_path, required_core)

    # Resolve redundant current channels once per shot (preshot stage).
    # Logs which channels are healthy/dead and raises if both of a pair are dead.
    from .current_channels import resolve_all
    ch_signals, ch_provenance = resolve_all(shot_path, discharge_current)
    It_arr  = ch_signals["IT"]
    Ioh_arr = ch_signals["IOH"]
    Iv_arr  = ch_signals["IV"]
    Ip_arr_resolved = ch_signals["IP"]
    print("[current_channels] resolved: "
          + "; ".join(f"{k}={v}" for k, v in ch_provenance.items()))

    # The resolved arrays share IP1's time base and its line ordering, so a line
    # counter into the file is a valid index into them.
    def _current_at(arr, line_idx):
        """Return resolved current value at line_idx, clamped to array length."""
        return float(arr[min(line_idx, len(arr) - 1)])

    #### Begin calculation ####
    # Only GBP and IP1 files are needed in the line-by-line loop; current pickup
    # is handled by the pre-resolved arrays above.
    loop_files_set = set(
        ["IP1.txt"] +
        ["GBP" + i + "T.txt" for i in unique_probes]
    )

    with ExitStack() as stack:
        # prepare all required files for reading [..., (file_i, file_handle_i),...]
        files = [(file_name, stack.enter_context(open(os.path.join(shot_path, file_name), "r"))) for file_name in loop_files_set]

        #find index of plasma current in files
        Ip_index = next(i for i, (name, _) in enumerate(files) if name == "IP1.txt")

        #create empty lists to store solution of all probes
        dR_sol = [[0]*(number_of_probe_set)]
        dZ_sol = [[0]*number_of_probe_set]
        # ADDED: collect fitted plasma current per set (only populated when the
        # M-probe estimator runs with fit_ip=True; stays empty otherwise).
        ip_sol = [[0]*number_of_probe_set]

        time_arr = [] #list to store time of each line
        IP_arr = [] #list to store plasma current of each line

        #threshold to help determine ending of discharge
        pass_threshold = False

        #count lines for tqdm
        num_lines = sum(1 for _ in files[0][1])
        files[0][1].seek(0)
        line_idx = 0   # tracks position in resolved current arrays
        for lines in tqdm(zip(*(f for _, f in files)), total=num_lines): # loop through lines simultaneously in all files

            
            # skip empty lines and header using IP1 as reference 
            if not lines[Ip_index].strip() or not lines[Ip_index].strip()[0].isdigit():
                line_idx += 1
                continue

            #conditions for discharge begin and end
            plasma_current = float(lines[Ip_index].strip().split()[1])
            # If plasma current lower than discharge_current threshold, then skip line
            if plasma_current < discharge_current:
                line_idx += 1
                continue
            # begin consideration of discharge end once plasma current pass through offset_threshold
            if plasma_current > discharge_current + discharge_offset: pass_threshold = True
            # stop calculation once plasma current drop below threshold
            # pass_threshold is important to disregard fluctuations in plasma current near discharge threshold
            elif plasma_current < discharge_current and pass_threshold: break

            ### Prepare signal at current line ###
            #create dictionary to keep all signals at current line
            raw_signal_dict = {}

            #add signal value from each file into signal_dict
            for (file_name, _), line in zip(files, lines):
                time, signal = line.split()
                raw_signal_dict[file_name.split(".")[0]] = float(signal) #use file name without .txt as keys
            raw_signal_dict["Time (ms)"] = float(time) #add time only once

            # Use resolved current values for this timestep.
            raw_signal_dict["IP"] = _current_at(Ip_arr_resolved, line_idx)
            It_val  = _current_at(It_arr,  line_idx)
            Ioh_val = _current_at(Ioh_arr, line_idx)
            Iv_val  = _current_at(Iv_arr,  line_idx)

            # correct magnetic signal from machine's noise
            corrected_signal_dict = correct_magnetic_signal(
                raw_signal_dict, It_val, Ioh_val, Iv_val)
            #######################################

            ### Calculate displacement value for current line ###
            time_arr.append(corrected_signal_dict["Time (ms)"])
            IP_arr.append(corrected_signal_dict["IP"])

            #calculation result from current line of data
            dR_line_sol = []
            dZ_line_sol = []
            ip_line_sol = []

            for index in range(number_of_probe_set):

                #retreive current set of probes to calculate and convert to list[int]
                probe_set = list(map(int, use_probe_set[index].split()))
                signal = [corrected_signal_dict[f"GBP{i}T"] for i in probe_set]

                #shift value at previous line
                # NOTE (2D): cal_shift does not use the previous-step estimate.
                # dR_prev/dZ_prev are passed for signature compatibility only and
                # are INERT -- there is no timestep recurrence in the 2D map.
                dR_prev, dZ_prev = dR_sol[-1][index],dZ_sol[-1][index]

                #restrict value of previous shift to be within shift_domain
                dR_prev = restrict_displacement(dR_prev, shift_domain)
                dZ_prev = restrict_displacement(dZ_prev, shift_domain)

                # ADDED (M-probe): when enabled, use the weighted M-probe
                # estimator with the measured plasma current (or fitted current
                # if fit_ip=True). The previous-step values remain unused.
                if mprobe_est is not None:
                    dR, dZ, _ip_used = mprobe_est[use_probe_set[index]].shift(
                        signal, corrected_signal_dict["IP"])
                    ip_line_sol.append(_ip_used)
                else:
                    #calculate dR and dZ (2D map: depends on current signal only)
                    ((dR, _),(dZ, _)) = cal_shift(DxDz_method=cal_dXdZ, taylor_order=taylor_order,
                                              signal=signal,est_horizontal_shift=dR_prev,
                                              est_vertical_shift=dZ_prev,probe_number=probe_set
                                              )
                    ip_line_sol.append(float("nan"))

                dR_line_sol.append(dR)
                dZ_line_sol.append(dZ)

            #add line result to final result
            dR_sol.append(dR_line_sol)
            dZ_sol.append(dZ_line_sol)
            ip_sol.append(ip_line_sol)
            line_idx += 1
            #########################################################

    #remove initial guess of 0 displacement (for matching dimension with time and plasma current)
    dR_sol.pop(0)
    dZ_sol.pop(0)
    ip_sol.pop(0)

    time_series = pd.Series(data = time_arr, name = "Time (ms)")
    IP_series = pd.Series(data = IP_arr, name = "IP (A)")
    dR_df = pd.DataFrame(data = dR_sol, columns=[probe_set + " R" for probe_set in use_probe_set])
    dZ_df = pd.DataFrame(data = dZ_sol, columns=[probe_set + " Z" for probe_set in use_probe_set])

    out = [time_series, IP_series, dR_df, dZ_df]
    # ADDED: expose fitted plasma current columns when the M-probe fit_ip mode ran.
    # Columns named "<set> Ifit"; in measured-Ip or non-mprobe mode these are NaN.
    if mprobe is not None and bool(mprobe.get("fit_ip", False)):
        ip_df = pd.DataFrame(data = ip_sol, columns=[probe_set + " Ifit" for probe_set in use_probe_set])
        out.append(ip_df)

    return pd.concat(out, axis = 1)

if __name__ == "__main__":
    use_probes = ["1 4 7 10", "2 4 8 10"]

    import pytest

    def assert_TFM_error(shot_path, use_probes_set):
        with pytest.raises(FileNotFoundError) as errinfo:
            TFM_main(shot_path,use_probes_set)
        #print(errinfo)

    assert_TFM_error(os.path.join("resources","fullShotData","1671"),use_probes)
    assert_TFM_error(os.path.join("resources","fullShotData","test_direc"),use_probes)

    displacement_df = TFM_main(os.path.join("resources","fullShotData","1641"),use_probes)

    fig, ax = plt.subplots(2)

    ax[0].plot(displacement_df["Time (ms)"], displacement_df["IP (A)"])
    ax[0].axhline(2500)
    
    ax[1].plot(displacement_df["Time (ms)"], displacement_df["1 4 7 10 R"])
    ax[1].plot(displacement_df["Time (ms)"], displacement_df["2 4 8 10 R"])
    ax[1].set_ylim(-0.3,0.3)
    plt.show()