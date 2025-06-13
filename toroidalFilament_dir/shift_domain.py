"""
Determine domain of plasma column position shift for each set of magnetic probe
Use the probe set with minimum domain as overall domain
"""

from .DxDz import cal_newton_DxDz as cal_DxDz
from .geometry_TT1 import all_arrays, coil_angle_dict
from .signal_strength import cal_signal

import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from tqdm import tqdm

def cal_Dx_arr(probe_set,R_shift_range,Z_shift):
    probe_angles = [coil_angle_dict[probe_num] for probe_num in probe_set]
    Dx_arr = []

    for R_shift in R_shift_range:
        probe_signal = cal_signal(R_shift, Z_shift, probe_angles)
        Dx, _ = cal_DxDz(probe_signal,probe_angles)

        Dx_arr.append(Dx)

    return Dx_arr

def find_R_shift_domain(Dx_arr,R_shift_range):
    """
    calculate domain of plasma shift for one set of magnetic probes
    """
    
    Dx_local_maxima_index, _ = find_peaks(Dx_arr)

    #determine nearest local maximas
    n = len(R_shift_range)
    # Safely get left and right domain indices
    left_candidates = Dx_local_maxima_index[Dx_local_maxima_index < n // 2]
    right_candidates = Dx_local_maxima_index[Dx_local_maxima_index > n // 2]

    left_domain_index = left_candidates.max() if len(left_candidates) > 0 else None
    right_domain_index = right_candidates.min() if len(right_candidates) > 0 else None


    left_domain, right_domain = R_shift_range[left_domain_index], R_shift_range[right_domain_index]
    if left_domain_index == None: left_domain = None
    if right_domain_index == None: right_domain = None

    return left_domain, right_domain

# Dx_arr = cal_Dx_arr([3,4,9,10],R_shift_range,Z_shift = 0.2)
# left_domain, right_domain = find_R_shift_domain(Dx_arr,R_shift_range)
# print(left_domain, right_domain)

R_shift_range = np.linspace(-0.3,0.3,501)
Z_shift_range = np.linspace(-0.1,0.1,51)
max_left, min_right = -np.inf, np.inf

for probe_set in tqdm(all_arrays):
    for Z_shift in Z_shift_range:

        Dx_arr = cal_Dx_arr(probe_set,R_shift_range,Z_shift)
        left_domain, right_domain = find_R_shift_domain(Dx_arr, R_shift_range)

        if left_domain != None:
            if left_domain > max_left: max_left = left_domain
        if right_domain != None:
            if right_domain < min_right: min_right = right_domain

print(max_left,min_right)