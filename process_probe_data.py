import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


"""
retreive and process experimental data for toroidal filament model
"""

path_plasma_current = os.getcwd() + r"\resources\magneticSignal\Plasma current for plasma position.xlsx"
path_magnetic_signal = os.getcwd() + r"\resources\magneticSignal\Magnetic probe GBP_T for plasma position.xlsx"

def discharge_duration(time, plasma_current) -> tuple:
    """
    Calculate discharge time from plasma current using maximum current as reference.
    
    Args:
        time: Array of recorded time values
        plasma_current: Array of recorded plasma current values
        
    Returns:
        Tuple of (discharge_begin, discharge_end) times
        
    Raises:
        ValueError: If no plasma current detected
    """
    # Find the index of maximum current
    max_current_index = np.argmax(plasma_current)
    max_current = plasma_current[max_current_index]
    
    if max_current < 1e-5:
        raise ValueError("No significant plasma current detected")
    
    # Find discharge begin (left of max current)
    begin_idx = max_current_index
    while begin_idx > 0 and plasma_current[begin_idx] > 1e-5:
        begin_idx -= 1
    
    # Find discharge end (right of max current)
    end_idx = max_current_index
    while end_idx < len(plasma_current) - 1 and plasma_current[end_idx] > 1e-5:
        end_idx += 1
    
    # Return the corresponding times
    return (time[begin_idx], time[end_idx])

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
    retreive magnetic signal and flip signs of probes facing in different directions

    :param shot_no: experimental shot number
    :return: data frame of corrected signal (magnetic_signal_df)
    """
    magnetic_signal_df = pd.read_excel(path_magnetic_signal, sheet_name = f"shot_{shot_no}")

    #one of the column has more data points
    min_len = magnetic_signal_df.dropna().shape[0]
    magnetic_signal_df = magnetic_signal_df.iloc[:min_len]

    #flip sign of probes facing different directions
    #all signals will be positive
    magnetic_signal_df.iloc[:,1:10] *= -1
    magnetic_signal_df.iloc[:,12] *= -1

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