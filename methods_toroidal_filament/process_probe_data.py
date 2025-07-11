import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


"""
retreive and process experimental data for toroidal filament model
"""

path_plasma_current = os.getcwd() + r"\resources\magneticSignal\Plasma current for plasma position.xlsx"
path_magnetic_signal = os.getcwd() + r"\resources\magneticSignal\Magnetic probe GBP_T for plasma position.xlsx"

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
    plasma_current_df = pd.read_excel(path_plasma_current, sheet_name = "Sheet1")

    recorded_time_df = plasma_current_df.loc[:, "Time [ms]"]
    recorded_plasma_current_df = plasma_current_df.loc[:,shot_no]

    start_discharge, end_discharge = discharge_duration(recorded_time_df, recorded_plasma_current_df)

    return recorded_plasma_current_df, recorded_time_df, start_discharge, end_discharge

def retreive_magnetic_signal(shot_no):
    """
    retreive magnetic signal from excel workbook

    :param shot_no: experimental shot number
    :return: data frame of corrected signal (magnetic_signal_df)
    """
    magnetic_signal_df = pd.read_excel(path_magnetic_signal, sheet_name = f"shot_{shot_no}")

    #one of the column has more data points
    min_len = magnetic_signal_df.dropna().shape[0]
    magnetic_signal_df = magnetic_signal_df.iloc[:min_len]

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
    trimmed_magnetic_signal_df = trimmed_magnetic_signal_df - trimmed_magnetic_signal_df.iloc[0]
    return trimmed_time_df.iloc[1:], trimmed_plasma_current_df.iloc[1:], trimmed_magnetic_signal_df.iloc[1:]