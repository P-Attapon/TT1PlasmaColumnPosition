import numpy as np
import scipy as sc
import warnings
from numpy.typing import NDArray
from .geometry_TT1 import R0, mu, I, R

"""
Calculate magnetic signal
"""
def coil_signal(phi:np.float64, r:float, z:float, a_f:float):
    """
    simulate magnetic signal at each coils in cylindrical coordinate

    Parameters:
    phi (float): angle between the positive x-axis and radial vector of the coil location in radian
    r (float) : radial distance from center of reactor to coil in meters
    z (float) : vertical distance from center of reactor to coil in meters
    a_f (float) : plasma radius in meters

    Returns:
    float: simulated magnetic signal at each coils
    """
    if None in (phi,r,z,a_f) or np.inf in (phi,r,z,a_f):
        raise ValueError(f"Nan or inf is contained in [phi,r,z,a_f] = {[phi,r,z,a_f]}")
    if a_f < 0:
        raise ValueError("a_f is negative")

    k = np.sqrt(4 * a_f * r / ((a_f + r) ** 2 + z ** 2))
    K, E = sc.special.ellipk(k ** 2), sc.special.ellipe(k ** 2)
    def b_r(r, z, a_f):  # calculate magnetic signal along radial direction with elliptic integrals
        return mu * I / 2 / np.pi * z / r / np.sqrt((a_f + r) ** 2 + z ** 2) * (
                -K + E * (a_f ** 2 + r ** 2 + z ** 2) / ((a_f - r) ** 2 + z ** 2)
                )

    def b_z(r, z, a_f):  # calculate magnetic signal along z direction with elliptic integrals
        return mu * I / 2 / np.pi * 1 / np.sqrt((a_f + r) ** 2 + z ** 2) * (
                K + E * (a_f ** 2 - r ** 2 - z ** 2) / ((a_f - r) ** 2 + z ** 2)
                )

    return abs(-b_r(r, z, a_f) * np.sin(phi) + b_z(r, z, a_f) * np.cos(phi)) #absolute because signal is always positive regardless of plasma current vector


def cal_signal(horizontal_shift: float, vertical_shift: float, coil_angle: NDArray[np.float64]):
    """
    calculate magnetic signal at each magnetic probe in given cross

    Parameters:
    horizontal_shift (float): plasma shift value along x/radial direction
    vertical_shift (float): plasma shift value along z/vertical direction
    coil_angle (list[float]): list of angle between the positive x-axis and radial vector each coil in the cross

    Returns:
    list[float]: magnetic signal at each coils
    """
    x, z = horizontal_shift, vertical_shift
    a_f:float = R0 + x  # calculate plasma radius from shift

    loc: list[tuple[float, float]] = [(R0 + R * np.cos(phi), R * np.sin(phi) - z) for phi in
                    coil_angle]  # get coordinate of each coil in cylindrical coordinate

    return [coil_signal(np.float64(coil_angle[i]), *loc[i], a_f) for i in range(len(coil_angle))]
