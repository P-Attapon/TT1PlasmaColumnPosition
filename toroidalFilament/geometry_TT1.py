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
    r, z = tup[0], tup[1]
    return np.arctan2(z,(r-R0 * 1000))

#array of angle of each magnetic probes, in radian
cross_perfect: list[NDArray[np.float64]] = [np.pi*np.array([0,1/2,1,3/2]) + i*np.pi/6 for i in [-1,0,1]]
cross: list[NDArray[np.float64]] = [np.array(list(map(probe_angle,[(925.8,-173.3), (822.3,296.1),(379.9,165.7),(483.3,-276.8)]))),
                                    np.array(list(map(probe_angle,[(973.9,-14.15),(663.1,317.3),(331.7,6.487),(642.5,-325)]))),
                                    np.array(list(map(probe_angle,[(936.1,147.8),(501.2,279.4),(369.5,-155.5),(807.6,-285.4)])))]

#IGBP dictionary of index in cross and IGBP probe number
coil_dict: dict[tuple[int]: int] = {
    (0,0) : 12, (0,1) : 3, (0,2) : 6, (0,3) : 9,
    (1,0) : 1, (1,1) : 4, (1,2) : 7, (1,3) : 10,
    (2,0) : 2, (2,1) : 5, (2,2) : 8, (2,3) : 11,
}