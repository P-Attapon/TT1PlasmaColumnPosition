from .geometry_TT1 import R
import scipy as sc
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

    if np.isinf(frac2): print(signal,np.rad2deg(c_angle))

    Dx = R/np.sin(phi1-phi2) * ( frac1*np.sin(phi2) - frac2*np.sin(phi1) )
    Dz = R/np.sin(phi1-phi2) * ( -frac1*np.cos(phi2) + frac2*np.cos(phi1) )

    return [Dx,Dz]

"""
Calculate plasma shift (DeltaX & DeltaZ) based on symbolic estimation of Dx & Dz

    (S_{i} - S_{i+pi})/(S_{i} - S_{i+pi}) = 1/R(Dz*sin(i) + Dx*cos(i))

    =>  Dx = R / np.cos(coil_angle[0]) * (S0 - S2) / (S0 + S2)
    =>  Dz = R / np.sin(coil_angle[1]) * (S1 - S3) / (S1 + S3)
"""

def cal_approx_DxDz(signal: list[float], coil_angle: NDArray[np.float64]) -> NDArray:
    """
    calculate Dx & Dz with rough symbolic approximation

    :param:
    M_angle (list[float]): list of angle between the positive x-axis and radial vector each coil in the cross

    :return:
    float: Dx, Dz in meters
    """
    S0, S1, S2, S3 = signal

    Dx = R / np.cos(coil_angle[0]) * (S0 - S2) / (S0 + S2)
    Dz = R / np.sin(coil_angle[1]) * (S1 - S3) / (S1 + S3)

    return np.array([Dx, Dz])

def cal_exact(signal:list[float], coil_angle = NDArray[np.float64]) -> NDArray:
    """
    directly calculate the value of Dx and Dz without any approximation

    :param signal: magnetic signal from each probe within the cross
    :param coil_angle: angle of the first coil within the cross
    :return: Dx and Dz
    """

    S0, S1, S2, S3 = signal
    phi0,phi1,phi2,phi3 = coil_angle

    def term(Dx,Dz,phi):
        nomi = R - (Dx*np.cos(phi) + Dz*np.sin(phi))
        denom = (R*np.cos(phi) - Dx)**2  + (R*np.sin(phi) - Dz)**2
        return nomi/denom

    def eqn(vars):
        Dx, Dz = vars

        #1st opposite pair
        term0,term2 = term(Dx,Dz,phi0), term(Dx,Dz,phi2)

        #2nd opposite pair
        term1,term3 = term(Dx,Dz,phi1), term(Dx,Dz,phi3)

        #equations to be solved
        eqn1 = (S0 - S2)/(S0 + S2) - (term0 - term2)/(term0 + term2)
        eqn2 = (S1 - S3)/(S1 + S3) - (term1 - term3)/(term1 + term3)

        return [eqn1,eqn2]

    q0 = np.array([0.00,0.00])
    result = sc.optimize.root(eqn, q0, method="hybr")
    return result.x