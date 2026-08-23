import os
import numpy as np
import pandas as pd
from .parameters import calibration_coeff

path_full_shot_directory = os.path.join("data")

"""
retreive and process experimental data for toroidal filament model
"""
def read_txt(file,columns):
    """
    read out first and second columns from text file data
    :param file: file location
    :param header: header of df
    :return: separated columns
    """
    df = pd.read_csv(file, sep = r"\s+", skiprows=8,header = None, names = columns)

    return df

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

def retreive_plasma_current(shot_no):
    """
    retreive plasma current along with time and calculate discharge time

    :param shot_no: experimental shot number
    :return: (recorded_plasma_current_df, recorded_time_df, start_discharge, end_discharge)
    """
    
    shot_no = str(shot_no)
    path = os.path.join(path_full_shot_directory,shot_no,"IP1.txt")
    df = read_txt(path, ["Time (ms)", "Ip (A)"])

    recorded_time_df = df["Time (ms)"]
    recorded_plasma_current_df = df["Ip (A)"]

    start_discharge, end_discharge = discharge_duration(recorded_time_df, recorded_plasma_current_df)
    return recorded_plasma_current_df, recorded_time_df, start_discharge, end_discharge

def retreive_magnetic_signal(shot_no):
    """
    retreive magnetic signal from excel workbook

    :param shot_no: experimental shot number
    :return: data frame of corrected signal (magnetic_signal_df)
    """
    probe_names = [f"GBP{i}T" for i in range(1,13)]
    headers = ["Time (ms)"] + probe_names

    magnetic_signal_df = pd.DataFrame(columns= headers)

    for probe in probe_names:
        file = os.path.join(path_full_shot_directory, str(shot_no), probe + ".txt")
        probe_signal = read_txt(file, ["Time (ms)", probe])

        magnetic_signal_df[probe] = probe_signal[probe]
        time = probe_signal["Time (ms)"]
    
    magnetic_signal_df["Time (ms)"] = time

    return magnetic_signal_df

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

    #remove noise using signal at t1
    # trimmed_magnetic_signal_df = trimmed_magnetic_signal_df - trimmed_magnetic_signal_df.iloc[0]
    return trimmed_time_df.iloc[1:], trimmed_plasma_current_df.iloc[1:], trimmed_magnetic_signal_df.iloc[1:]

################ Signal Calibration ################
def noise_value(kt,It,koh,Ioh,kv,Iv):
    return kt * It + koh * Ioh + kv * Iv

def mk_noise_df(It,Ioh,Iv,num_col = 13):
    columns = {"Time (ms)":[]}
    for i in range(1,12):
        columns[f"GBP{i}T"] = []
    
    noise_df = pd.DataFrame(columns)
    noise_df["Time (ms)"] = It.iloc[:,0]

    It_np = It.to_numpy()[:,1]
    Ioh_np = Ioh.to_numpy()[:,1]
    Iv_np = Iv.to_numpy()[:,1]

    for probe_number in range(1, num_col):
        kt = calibration_coeff[f"k{probe_number}t"]
        koh = calibration_coeff[f"k{probe_number}oh"]
        kv = calibration_coeff[f"k{probe_number}v"]

        noise_df.loc[:, f"GBP{probe_number}T"] = noise_value(
            kt, It_np, koh, Ioh_np, kv, Iv_np
        )

    return noise_df

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

    return raw_B - kt * It - koh * Ioh - kv * Iv

def calibrate_signal_df(plasma_signal,It,Ioh,Iv):
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

    num_row, num_col = plasma_signal.shape
    calibrated_signal_np = plasma_signal.to_numpy()
    plasma_signal_np = plasma_signal.to_numpy()
    
    It_np = It.to_numpy()
    Ioh_np = Ioh.to_numpy()
    Iv_np = Iv.to_numpy()

    for probe_number in range(1, num_col):
        kt = calibration_coeff[f"k{probe_number}t"]
        koh = calibration_coeff[f"k{probe_number}oh"]
        kv = calibration_coeff[f"k{probe_number}v"]

        calibrated_signal_np[:, probe_number] = magnetic_field_calibration(
            plasma_signal_np[:, probe_number], kt, It_np, koh, Ioh_np, kv, Iv_np
        )

    calibrated_signal_df = pd.DataFrame(calibrated_signal_np, columns=plasma_signal.columns)

    return calibrated_signal_df