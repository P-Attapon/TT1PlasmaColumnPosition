import numpy as np
import pandas as pd
import os
from contextlib import ExitStack
from tqdm import tqdm
from matplotlib import pyplot as plt

from methods_script.toroidal_filament.parameters import all_arrays, shift_domain
from methods_script.toroidal_filament.plasma_shift import cal_shift
from methods_script.toroidal_filament.DxDz import cal_newton_DxDz as cal_dXdZ
from methods_script.toroidal_filament.process_probe_data import magnetic_field_calibration, calibration_coeff

taylor_order = 3 #order of taylor series fitting => must match with fitting coefficient file!

def determine_unique_probes(probe_set:list[str]) -> list[str]:
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

def correct_magnetic_signal(signal_dict: dict) -> dict:
    """remove noise from probe's signal"""
    corrected_signal = {key: None for key in signal_dict.keys() if "GBP" in key}

    for key in corrected_signal.keys():
        probe_num = key[3:5] if key[3:5].isnumeric() else key[3]
        kt, koh, kv = calibration_coeff["k"+ probe_num +"t"], calibration_coeff["k"+ probe_num +"oh"], calibration_coeff["k"+ probe_num +"v"]
        It, Ioh, Iv = signal_dict["IT1"], signal_dict["IOH1"], signal_dict["IV2"]
        corrected_signal[key] = magnetic_field_calibration(
            signal_dict[key],kt,It,koh,Ioh,kv,Iv
        )

    corrected_signal["Time (ms)"] = signal_dict["Time (ms)"]
    corrected_signal["IP1"] = signal_dict["IP1"]
    return corrected_signal

def restrict_displacement(displacement_val, shift_domain):
    if abs(displacement_val) > shift_domain:
        if displacement_val < -shift_domain: displacement_val = -shift_domain
        elif displacement_val > shift_domain: displacement_val = shift_domain
    return displacement_val


def TFM_main(shot_path: str,use_probe_set: list[str],discharge_current:float=2500, discharge_offset: float = 100) -> pd.DataFrame:
    """
    Calculate plasma column position displacement with the Toroidal Filament Model
    :param shot_path: path to data directory containing
                      "IP1.txt", "IT1.txt", "IOH1.txt", "IV2.txt", and
                      "GBPXT.txt" for all X as unique probe number in probe_set
    :param use_probe_set: set of magnetic probes to use for calculation
                      (set number must exist in all_arrays)
    :param discharge_current: threshold for begin and end of discharge
    :param discharge_offset: constant offset value helps to determine ending of discharge
    :return: dataframes of centroid displacement calculated from all probe_set
             along radial and vertical directions
    """
    number_of_probe_set = len(use_probe_set)
    #determine all unique magnetic probes to use
    unique_probes = determine_unique_probes(use_probe_set)

    #define all required files
    required_files = set(
        ["IP1.txt", "IT1.txt", "IOH1.txt", "IV2.txt"] + 
        ["GBP" + i + "T.txt" for i in unique_probes]
    )

    #raise FileNotFoundError if shot_path does not exist or
    #missing required txt files
    check_missing_files(shot_path, required_files)

    #### Begin calculation ####

    with ExitStack() as stack:
        # prepare all required files for reading [..., (file_i, file_handle_i),...]
        files = [(file_name, stack.enter_context(open(os.path.join(shot_path, file_name), "r"))) for file_name in required_files]

        #find index of plasma current in files
        Ip_index = next(i for i, (name, _) in enumerate(files) if name == "IP1.txt")

        #create empty lists to store solution of all probes
        dR_sol = [[0]*(number_of_probe_set)]
        dZ_sol = [[0]*number_of_probe_set]

        time_arr = [] #list to store time of each line
        IP_arr = [] #list to store plasma current of each line

        #threshold to help determine ending of discharge
        pass_threshold = False

        #count lines for tqdm
        num_lines = sum(1 for _ in files[0][1])
        files[0][1].seek(0)
        for lines in tqdm(zip(*(f for _, f in files)), total=num_lines): # loop through lines simultaneously in all files

            
            # skip empty lines and header using IP1 as reference 
            if not lines[Ip_index].strip() or not lines[Ip_index].strip()[0].isdigit(): continue

            #conditions for discharge begin and end
            plasma_current = float(lines[Ip_index].strip().split()[1])
            # If plasma current lower than discharge_current threshold, then skip line
            if plasma_current < discharge_current: continue
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

            # correct magnetic signal from machine's noise
            corrected_signal_dict = correct_magnetic_signal(raw_signal_dict)
            #######################################

            ### Calculate displacement value for current line ###
            time_arr.append(corrected_signal_dict["Time (ms)"])
            IP_arr.append(corrected_signal_dict["IP1"])

            #calculation result from current line of data
            dR_line_sol = []
            dZ_line_sol = []

            for index in range(number_of_probe_set):

                #retreive current set of probes to calculate and convert to list[int]
                probe_set = list(map(int, use_probe_set[index].split()))
                signal = [corrected_signal_dict[f"GBP{i}T"] for i in probe_set]

                #shift value at previous line
                dR_prev, dZ_prev = dR_sol[-1][index],dZ_sol[-1][index]

                #restrict value of previous shift to be within shift_domain
                dR_prev = restrict_displacement(dR_prev, shift_domain)
                dZ_prev = restrict_displacement(dZ_prev, shift_domain)

                ((dR, _),(dZ, _)) = cal_shift(DxDz_method=cal_dXdZ, taylor_order=taylor_order,
                                              signal=signal,est_horizontal_shift=dR_prev,
                                              est_vertical_shift=dZ_prev,probe_number=probe_set
                                              )
                
                dR_line_sol.append(dR)
                dZ_line_sol.append(dZ)
            
            #add line result to final result

            dR_sol.append(dR_line_sol)
            dZ_sol.append(dZ_line_sol)
            #########################################################

    #remove initial guess of 0 displacement (for matching dimension with time and plasma current)
    dR_sol.pop(0)
    dZ_sol.pop(0)

    time_series = pd.Series(data = time_arr, name = "Time (ms)")
    IP_series = pd.Series(data = IP_arr, name = "IP (A)")
    dR_df = pd.DataFrame(data = dR_sol, columns=[probe_set + " R" for probe_set in use_probe_set])
    dZ_df = pd.DataFrame(data = dZ_sol, columns=[probe_set + " Z" for probe_set in use_probe_set])

    return pd.concat([time_series, IP_series, dR_df, dZ_df], axis = 1)

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