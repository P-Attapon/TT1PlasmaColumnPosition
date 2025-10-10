import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .parameters import calibration_coeff

"""
retreive and process experimental data for toroidal filament model
"""

path_plasma_current = os.path.join(os.getcwd(),"resources","magneticSignal","Plasma current for plasma position.xlsx")
path_magnetic_signal = os.path.join(os.getcwd(),"resources","magneticSignal","Magnetic probe GBP_T for plasma position.xlsx")
path_full_shot_directory = os.path.join(os.getcwd(), "resources", "fullShotData")

##### Data acquisition #####

def read_txt(file):
    """
    read out first and second columns from text file data
    :param file: file location
    :return: separated columns
    """
    df = pd.read_csv(file, sep = "\s+", skiprows=8,header = None)

    col1 = df.iloc[:,0]
    col2 = df.iloc[:,1]

    return col1, col2

def retrieve_plasma_current(shot_no, is_excel = True):
    """
    retrieve plasma current along with time and calculate discharge time

    :param shot_no: experimental shot number
    :param is_excel: specify if data is stored in excel workbook in magneticSignal folder or fullShotData
    :return: (recorded_plasma_current_df, recorded_time_df, start_discharge, end_discharge)
    """

    if is_excel:

        plasma_current_df = pd.read_excel(path_plasma_current, sheet_name = "Sheet1")

        recorded_time_df = plasma_current_df.loc[:, "Time [ms]"]
        recorded_plasma_current_df = plasma_current_df.loc[:,shot_no]

    else:
        shot_no = str(shot_no)
        path = os.path.join(path_full_shot_directory,shot_no,"IP1.txt")
        recorded_time_df, recorded_plasma_current_df = read_txt(path)

    start_discharge, end_discharge = discharge_duration(recorded_time_df, recorded_plasma_current_df)
    return recorded_plasma_current_df, recorded_time_df, start_discharge, end_discharge

def retrieve_magnetic_signal(shot_no,is_excel = True):
    """
    retrieve magnetic signal

    :param shot_no: experimental shot number
    :param is_excel: specify if data is stored in excel workbook in magneticSignal folder or fullShotData
    :return: data frame of corrected signal (magnetic_signal_df)
    """

    if is_excel:
        magnetic_signal_df = pd.read_excel(path_magnetic_signal, sheet_name = f"shot_{shot_no}")

        #one of the column has more data points
        min_len = magnetic_signal_df.dropna().shape[0]
        magnetic_signal_df = magnetic_signal_df.iloc[:min_len]

    else:
        probe_names = [f"GBP{i}T" for i in range(1,13)]
        headers = ["Time (ms)"] + probe_names

        magnetic_signal_df = pd.DataFrame(columns= headers)

        for probe in probe_names:
            file = os.path.join(path_full_shot_directory, str(shot_no), probe + ".txt")
            time, magnetic_signal = read_txt(file)

            magnetic_signal_df[probe] = magnetic_signal
        
        magnetic_signal_df["Time (ms)"] = time

    return magnetic_signal_df

###########


##### Calculation from raw data #####
def discharge_duration(time, plasma_current,Ip_threshold = 2500) -> tuple:
    """
    Calculate discharge time from plasma current using maximum current as reference.
    
    Args:
        time: Array of recorded time values
        plasma_current: Array of recorded plasma current values
        Ip_threshold: thereshold of current to be considered plasma discharge
        
    Returns:
        Tuple of (discharge_begin, discharge_end) times
        
    Raises:
        ValueError: If no plasma current detected
    """
    time_begin, time_end = None, None

    #search from left to determine discharge begin
    for t, Ip in zip(time, plasma_current):

        if Ip >= Ip_threshold:
            time_begin = t
            break

    #search from right to determine discharge end
    for t, Ip in zip(reversed(time),reversed(plasma_current)):
        if Ip >= Ip_threshold:
            time_end = t
            break

    if time_begin == time_end: raise ValueError("No plasma discharge")
    
    return time_begin,time_end

def trim_quantities(recorded_time_df,magnetic_signal_df,recorded_plasma_current_df,t1,t2):
    """
    trim data frame of magnetic signal, time, and plasma current to be within desired time 
    and removing signal noise using signal at t1

    :param recorded_time_df: data frame of recorded time
    :param magnetic_signal_df: data frame of magenetic signal
    :param recorded_plasma_current_df: data frame of recorded plasma_current
    :param t1: initial time interval
    :param t2: final time interval
    :return: trimmed magnetic signal within t1 & t2 (trimmed_time_df, trimmed_plasma_current_df, trimmed_magnetic_signal_df)
    """
    #trim time data frame
    trimmed_time_df = recorded_time_df[(recorded_time_df > t1) & (recorded_time_df < t2)]

    #trim plasma current data frame
    trimmed_plasma_current_df = recorded_plasma_current_df[(recorded_time_df > t1) & (recorded_time_df < t2)]

    #extract region within interval
    trimmed_magnetic_signal_df = magnetic_signal_df[(magnetic_signal_df["Time (ms)"] > t1) & (magnetic_signal_df["Time (ms)"] < t2)]

    return trimmed_time_df.iloc[1:], trimmed_plasma_current_df.iloc[1:], trimmed_magnetic_signal_df.iloc[1:]

###############################################

#signal calibration

def magnetic_field_calibration(raw_B, kt, It, koh, Ioh, kv, Iv):
    """
    correct magnetic noise using calibration factors and machine's current
    :param raw_B: raw magnetic signal recorded from the magnetic probes
    :param kt: toroidal correction coefficient
    :param It: toroidal field current
    :param koh: ohmic correction coefficient
    :param Ioh: ohmic heating current
    :param kv: vertical field correction
    :param Iv: vertical field current
    :return: corrected magnetic field
    """
    return raw_B - (kt * It + koh * Ioh +  kv * Iv)

def calibrate_signal_df(plasma_signal, raw_time, discharge_begin,discharge_end,It,Ioh,Iv):
    """
    calibrate raw signal dataframe using magnetic_field_calibration
    :param plasma_signal: dataframe of raw magnetic signal
    :raw_time: time steps recorded by It, Ioh, and Iv
    :discharge_begin: begin time stamp of discharge
    :discharge_end: end time stamp of discharge
    :It: array of toroidal current (corresponding to raw_time)
    :Ioh: array of ohmic current (corresponding to raw_time)
    :Iv: array of vertical field current (corresponding to raw_time)
    :return: pandas dataframe of calibrated magnetic signal
    """
    discharge_mask = (raw_time > discharge_begin) & (raw_time < discharge_end)

    It = It[discharge_mask]
    Ioh = Ioh[discharge_mask]
    Iv = Iv[discharge_mask]

    len_of_plasma_signal, number_of_probes = plasma_signal.shape

    calibrated_signal_df = [None]*len_of_plasma_signal
    for i in range(len_of_plasma_signal):
        new_row = []
        for probe_number in range(1,number_of_probes):
            kt, koh, kv = calibration_coeff[f"k{probe_number}t"], calibration_coeff[f"k{probe_number}oh"], calibration_coeff[f"k{probe_number}v"]
            new_row.append(
                magnetic_field_calibration(plasma_signal.iloc[i,probe_number],kt,It.iloc[i],koh,Ioh.iloc[i],kv,Iv.iloc[i])
                )
            
        calibrated_signal_df[i] = new_row

    return pd.DataFrame(calibrated_signal_df)