from .parameters import R
import numpy as np
from numpy.typing import NDArray

"""
Calculate plasma shift (DeltaX & DeltaZ) based on numerical computation of Dx & Dz

    Use multi-dimensional Newton's method to solve for Dx and Dz
    (S_{i} - S_{i+pi})/(S_{i} - S_{i+pi}) = 1/R(Dz*sin(i) + Dx*cos(i))
"""
def cal_newton_DxDz(signal: list[float], c_angle: NDArray[np.float64]) -> list[float]:
    """
    calculate Dx&Dz using matrix form

    Parameters:
    signal (NDArray[np.float64]): array of signal measured in each manetic probe
    c_angle (NDArray[np.float64]): array of angle of each magnetic probe with respect to the x-axis
    """
    phi1, phi2 = c_angle[0], c_angle[1]
    frac1, frac2 = (signal[0] - signal[2])/(signal[0] + signal[2]), (signal[1] - signal[3])/(signal[1] + signal[3])

    Dx = -R/np.sin(phi1-phi2) * ( frac1*np.sin(phi2) - frac2*np.sin(phi1) )
    Dz = -R/np.sin(phi1-phi2) * ( -frac1*np.cos(phi2) + frac2*np.cos(phi1) )

    return [Dx,Dz]