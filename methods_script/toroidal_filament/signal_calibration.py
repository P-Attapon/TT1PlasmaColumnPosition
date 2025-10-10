import pandas as pd
from .parameters import calibration_coeff

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
    discharge_mask = (raw_time >= discharge_begin) & (raw_time <= discharge_end)

    It = It[discharge_mask]
    Ioh = Ioh[discharge_mask]
    Iv = Iv[discharge_mask]

    len_of_plasma_signal, number_of_probes = plasma_signal.shape
    calibrated_signal_df = [None]*len_of_plasma_signal
    for i in range(len_of_plasma_signal):
        new_row = []
        for j in range(number_of_probes):
            probe_number = j + 1
            kt, koh, kv = calibration_coeff[f"k{probe_number}t"], calibration_coeff[f"k{probe_number}oh"], calibration_coeff[f"k{probe_number}v"]
            new_row.append(
                magnetic_field_calibration(plasma_signal[i,j],kt,It[i],koh,Ioh[i],kv,Iv[i])
                )
            
        calibrated_signal_df[i] = new_row
    
    return pd.DataFrame(calibrate_signal_df)