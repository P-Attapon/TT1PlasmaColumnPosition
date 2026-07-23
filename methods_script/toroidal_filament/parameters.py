from scipy import constants
from numpy.typing import NDArray
import numpy as np

"""
Dimensions of TT1 cross-section
"""

base_decimal_precision = 3 #decimal precision of shift value in coefficient dictionary
shift_domain = 0.10 #[m]

R0: float = 0.65 #Major radius m
R:float = 0.321 #Minor radius of torus center to magnetic probes

mu: float = constants.mu_0 #magnetic permittivity constant
I: float = 100000.0 #Plasma current A 
#I is used to simulate magnetic field at each probe 
#final relationship of \Delta_R \propto D_x does not depends on I

# magnetic signal calibration coefficient calculated by A. Wisitsorasak
# kit: noise from toroidal field coil, 
# kioh: noise from ohmic heating, 
# kiv: noise from vertical field coils
calibration_coeff = {
    "k1t": 1.30033E-07, "k2t": 2.69785E-06, "k3t": 4.59026E-06, "k4t": 7.44631E-06, "k5t": 7.72376E-06, "k6t": -2.39202E-06, "k7t": -6.88199E-07, "k8t": -5.59296E-06, "k9t": -6.88533E-06, "k10t": -7.66234E-06, "k11t": 2.92252E-05, "k12t": 1.2445E-05,
    "k1oh": -1.47234E-07, "k2oh": 4.96741E-10, "k3oh": 1.94899E-08, "k4oh": -2.94754E-07, "k5oh": 4.49477E-07, "k6oh": -3.34029E-08, "k7oh": 7.16693E-07, "k8oh":-6.05933E-07, "k9oh": 8.83113E-08, "k10oh": -2.43559E-07, "k11oh":-8.05395E-07, "k12oh":-5.145E-07,
    "k1v": 9.82938E-06, "k2v": 1.16725E-05, "k3v": 1.18725E-05, "k4v": 3.00285E-06, "k5v": -8.25369E-06, "k6v": 4.32704E-06, "k7v": -1.79673E-05, "k8v": -1.5361E-05, "k9v": -3.64008E-06, "k10v": 4.24555E-06, "k11v": 2.49106E-05, "k12v":3.95202E-06
}


def probe_angle(tup):
    r, z = tup
    return np.arctan2(z,(r-R0 * 1000))

#array of angle of each magnetic probes, in radian

# theoretical polar angle of each probe on poloidal plane (without deviation from installation)
cross_perfect: list[NDArray[np.float64]] = [np.pi*np.array([0,1/2,1,3/2]) + i*np.pi/6 for i in [-1,0,1]]
cross: list[NDArray[np.float64]] = [np.array(list(map(probe_angle,[(925.8,-173.3), (822.3,296.1),(379.9,165.7),(483.3,-276.8)]))),
                                    np.array(list(map(probe_angle,[(973.9,-14.15),(663.1,317.3),(331.7,6.487),(642.5,-325)]))),
                                    np.array(list(map(probe_angle,[(936.1,147.8),(501.2,279.4),(369.5,-155.5),(807.6,-285.4)])))]

# measured polar angle of each probe on poloidal plane
coil_angle_dict = {
    1: probe_angle((973.9, -14.15)),
    2: probe_angle((936.1, 147.8)),
    3: probe_angle((822.3, 296.1)),
    4: probe_angle((663.1, 317.3)),
    5: probe_angle((501.2, 279.4)),
    6: probe_angle((379.9, 165.7)),
    7: probe_angle((331.7, 6.487)),
    8: probe_angle((369.5, -155.5)),
    9: probe_angle((483.3, -276.8)),
    10: probe_angle((642.5, -325)),
    11: probe_angle((807.6, -285.4)),
    12: probe_angle((925.8, -173.3))
}

# all possible sets of probes that can be used for calculation
all_arrays = [[11, 12, 5, 6], [11, 1, 5, 7], [11, 2, 5, 8], [11, 3, 5, 9], [11, 4, 5, 10], [12, 1, 6, 7], [12, 2, 6, 8],
              [12, 3, 6, 9], [12, 4, 6, 10], [1, 2, 7, 8], [1, 3, 7, 9], [1, 4, 7, 10], [2, 3, 8, 9], [2, 4, 8, 10], [3, 4, 9, 10]]

def probe_lst_to_str(lst):
    """
    convert list of probe numbers into keys of coefficients dictionary

    :param lst: list of 4 probe numbers
    :return: a key as string of input of coefficients map
    """
    arr_str = ""
    for i, probe_num in enumerate(lst): 
        arr_str += str(probe_num) if i==0 else " " + str(probe_num)
    return arr_str

error_dict = {
    probe_lst_to_str([11,12,5,6]) + "R": 8.398081e-03,
    probe_lst_to_str([11,1,5,7]) + "R": 0.008264,
    probe_lst_to_str([11,2,5,8]) + "R": 0.004972,
    probe_lst_to_str([11,3,5,9]) + "R": 0.003290,
    probe_lst_to_str([11,4,5,10]) + "R": 2.401759e-03,
    probe_lst_to_str([12,1,6,7]) + "R": 0.011363,
    probe_lst_to_str([12,2,6,8]) + "R": 0.01067944,
    probe_lst_to_str([12,3,6,9]) + "R": 0.007471,
    probe_lst_to_str([12,4,6,10]) + "R": 0.002210,
    probe_lst_to_str([1,2,7,8]) + "R": 0.018785,
    probe_lst_to_str([1,3,7,9]) + "R": 0.011572,
    probe_lst_to_str([1,4,7,10]) + "R": 2e-03,          # already in original
    probe_lst_to_str([2,3,8,9]) + "R": 0.002631736,
    probe_lst_to_str([2,4,8,10]) + "R": 10e-3,          # already in original
    probe_lst_to_str([3,4,9,10]) + "R": 0.002609,

    probe_lst_to_str([11,12,5,6]) + "Z": 0.010126,
    probe_lst_to_str([11,1,5,7]) + "Z": 1.100149e-03,
    probe_lst_to_str([11,2,5,8]) + "Z": 4.156911e-03,
    probe_lst_to_str([11,3,5,9]) + "Z": 0.003804,
    probe_lst_to_str([11,4,5,10]) + "Z": 0.008803185,
    probe_lst_to_str([12,1,6,7]) + "Z": 0.001388,
    probe_lst_to_str([12,2,6,8]) + "Z": 0.000679,
    probe_lst_to_str([12,3,6,9]) + "Z": 0.004467138,
    probe_lst_to_str([12,4,6,10]) + "Z": 0.010728,
    probe_lst_to_str([1,2,7,8]) + "Z": 0.005613,
    probe_lst_to_str([1,3,7,9]) + "Z": 0.003670659,
    probe_lst_to_str([1,4,7,10]) + "Z": 2e-03,           # already in original
    probe_lst_to_str([2,3,8,9]) + "Z": 0.012130,
    probe_lst_to_str([2,4,8,10]) + "Z": 2e-3,           # already in original
    probe_lst_to_str([3,4,9,10]) + "Z": 0.010763,
}
