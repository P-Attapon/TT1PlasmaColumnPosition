from scipy import constants
from numpy.typing import NDArray
import numpy as np

"""
Dimensions of TT1 cross-section
"""

R0: float = 0.65 #Major radius m
mu: float = constants.mu_0 #magnetic permittivity constant
I: float = 100000.0 #Plasma current A
R:float = 0.321 #Radial distance from cross-sectional center to each coil m

def probe_angle(tup):
    r, z = tup
    return np.arctan2(z,(r-R0 * 1000))

#array of angle of each magnetic probes, in radian
cross_perfect: list[NDArray[np.float64]] = [np.pi*np.array([0,1/2,1,3/2]) + i*np.pi/6 for i in [-1,0,1]]
cross: list[NDArray[np.float64]] = [np.array(list(map(probe_angle,[(925.8,-173.3), (822.3,296.1),(379.9,165.7),(483.3,-276.8)]))),
                                    np.array(list(map(probe_angle,[(973.9,-14.15),(663.1,317.3),(331.7,6.487),(642.5,-325)]))),
                                    np.array(list(map(probe_angle,[(936.1,147.8),(501.2,279.4),(369.5,-155.5),(807.6,-285.4)])))]

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