import os
import pandas as pd
from scipy.signal import find_peaks


"""
retreive and process experimental data for toroidal filament model
"""

path_plasma_current = os.getcwd() + r"\resources\magneticSignal\Plasma current for plasma position.xlsx"
path_magnetic_signal = os.getcwd() + r"\resources\magneticSignal\Magnetic probe GBP_T for plasma position.xlsx"

def discharge_duration(time, plasma_current) -> float:
    """
    calculate discharge time from plasma current

    :param time: array of recorded time
    :param plasma_current: array of recorded plasma current
    :return: discharge time
    """
    inverted_signal = -plasma_current
    peaks, _ = find_peaks(inverted_signal,height=4000,distance = 150)

    if len(peaks) == 0: raise ValueError("no peaks plasma current")
    current = plasma_current[peaks[0]]
    count = 0

    #find time for start of plasma discharge
    while current <= 0: #current equal to 0 will cause zero division
        count += 1
        current = plasma_current[peaks[0] + count]

    discharge_begin = time[peaks[0] + count]

    #find time for end of plasma discharge

    max_current = max(plasma_current) #the time must pass through the maximum current first to prevent fluctuation near discharge time
    pass_max = False

    while current > 0 or not pass_max:
        count += 1
        current = plasma_current[peaks[0] + count]

        if current == max_current: pass_max = True
    
    discharge_end = time[peaks[0] + count]

    return (discharge_begin, discharge_end)

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