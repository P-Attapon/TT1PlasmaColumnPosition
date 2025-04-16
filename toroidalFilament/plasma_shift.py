import numpy as np
import scipy as sc
from uncertainties import ufloat
from numpy.typing import NDArray
from .signal_strength import cal_signal
from typing import Callable

"""
Calculate alpha and beta for curve fitting
"""
# power series for curve fitting
def make_taylor(order,a) -> Callable:
    def taylor_polynomial(x,*coeff):
        return sum([coeff[i] * (x-a)**i for i in range(order + 1)])
    return taylor_polynomial

def cal_alpha(DxDz_method: Callable,a:float,taylor_order:int ,horizontal_shift: float,vertical_shift:float, coil_angle: NDArray[np.float64],
            vertical_range: NDArray[np.float64] = np.linspace(-0.05,0.05,101)) -> tuple:
    """
    calculate alpha coefficient from curve fitting

    :param:
    DxDz_method (Callable): method for calculation of Dx & Dz (cal_newton_DxDz/cal_approx_DxDz)
    :param Dz: point to be evaluate in taylor series
    horizontal_shift (float): plasma shift value along x/radial direction
    M_angle (list[float]): list of angle between the positive x-axis and radial vector each coil in the cross
    vertical_range (NDArray[np.float64): possible shift of plasma along vertical axis

    :return:
    array: optimal value of alpha from curve fitting
    matrix:covariance of alpha
    """
    #calculate Dz for each DeltaX & DeltaZ
    Dz = [DxDz_method(cal_signal(horizontal_shift, z_shift, coil_angle), coil_angle)[1] for z_shift in vertical_range]

    taylor_polynomial = make_taylor(taylor_order, a = a)
    #p0 -> taylor order + 1 because order n has n+1 coeff
    return sc.optimize.curve_fit(taylor_polynomial, Dz, vertical_range,p0 = [0.001]*(taylor_order+1),full_output=False,gtol=1e-8, xtol=1e-8)

def cal_beta(DxDz_method: Callable,a: float,taylor_order:int,vertical_shift: float,horizontal_shift:float, coil_angle: NDArray[np.float64],
             horizontal_range: NDArray[np.float64] = np.linspace(-0.05,0.05,101)) -> tuple:
    """
    calculate beta coefficient from curve fitting

    :param:
    DxDz_method (Callable): method for calculation of Dx & Dz (cal_newton_DxDz/cal_approx_DxDz) \n
    :param Dx: point to be estimated on the taylor's series
    taylor_order (int): degree of taylor's polynomial (start from 0)
    vertical_shift (float): plasma shift value along z/vertical direction \n
    M_angle (list[float]): list of angle between the positive x-axis and radial vector each coil in the cross \n
    horizontal_range (NDArray[np.float64): possible shift of plasma along horizontal axis \n

    :return:
    array: optimal value of beta from curve fitting \n
    matrix: covariance of beta
    """
    #calculate Dx for each DeltaX & DeltaZ
    Dx = [DxDz_method(cal_signal(x_shift, vertical_shift, coil_angle), coil_angle)[0] for x_shift in horizontal_range]

    taylor_polynomial = make_taylor(taylor_order, a = a)
    #p0 -> taylor order + 1 because order n has n+1 coeff
    return sc.optimize.curve_fit(taylor_polynomial, Dx, horizontal_range,p0 = [0.001] * (taylor_order+1),full_output=False,gtol=1e-8, xtol=1e-8)


def cal_shift(DxDz_method: Callable, taylor_order:int,signal: list[float], est_horizontal_shift: float, est_vertical_shift: float, coil_angle: NDArray[np.float64],
              alpha_vertical_range = np.linspace(-0.05,0.05,101), beta_horizontal_range = np.linspace(-0.05,0.05,101)) -> NDArray:
    """
    calculate DeltaX and DeltaZ from power series of Dx & Dz based on estimation of shift value

    :param:
    DxDz_method (Callable): method of calculation for Dx and Dz
    taylor_order (int): order of taylor's polynomial starting from 0
    signal (list[float]): magnetic signal at each magnetic probe in coil_angle
    est_horizontal_shift (float): estimate shift along x/radial direction
    est_vertical_shift (float): estimate shift along z/vertical direction
    coil_angle (list[float]): list of angle between the positive x-axis and radial vector each coil in the cross

    :return:
    matrix: [[x_shift, x_shift_uncertainty],
             [z_shift, z_shift_uncertainty]]
    """

    Dx, Dz = DxDz_method(signal, coil_angle)

    alpha, a_cov = cal_alpha(DxDz_method,Dz,taylor_order,est_vertical_shift,est_horizontal_shift, coil_angle,vertical_range = alpha_vertical_range)
    beta, b_cov = cal_beta(DxDz_method,Dx,taylor_order,est_horizontal_shift,est_vertical_shift, coil_angle, horizontal_range= beta_horizontal_range)

    a_cov, b_cov = np.diag(a_cov), np.diag(b_cov)

    vertical_shift = make_taylor(taylor_order, a = Dz)(Dz,*alpha)
    horizontal_shift = make_taylor(taylor_order, a = Dx)(Dx,*beta)

    vertical_shift_uncertainty = np.sqrt(np.sum((np.array(alpha) * a_cov) ** 2))
    horizontal_shift_uncertainty = np.sqrt(np.sum((np.array(beta) * b_cov) ** 2))

    return np.array([
        [horizontal_shift, horizontal_shift_uncertainty],
        [vertical_shift, vertical_shift_uncertainty]
    ])

"""
Combine DeltaX and DeltaZ from different crosses
"""
#Combining results from different pair of coils
def cal_combined_shift(DxDz_method: Callable, taylor_order:int,signal: list[list[float]], est_horizontal_shift: float, est_vertical_shift:
                        float, crosses: list[NDArray[np.float64]]) -> NDArray:
    """
    calculate mean DeltaX and DeltaZ from power series of Dx & Dz from every cross

    :param:
    signal (list[float]): magnetic signal at each magnetic probe in coil_angle
    est_horizontal_shift (float): estimate shift along x/radial direction
    est_vertical_shift (float): estimate shift along z/vertical direction
    coil_angle (list[float]): list of angle between the positive x-axis and radial vector each coil in the cross

    :return:
        NDArray[NDArray[np.float]] : mean plasma shift
        
        output format:
        matrix: [[x_shift, x_shift_uncertainty],
                [z_shift, z_shift_uncertainty]]
    """

    # calculate plasma shift for each cross
    shift = np.array([cal_shift(DxDz_method,taylor_order,signal[index], est_horizontal_shift, est_vertical_shift, crosses[index])[:, 0]
         for index in range(len(crosses))])

    #return array of mean shift
    return np.array([
        [np.mean(shift[:, 0]), np.std(shift[:, 0], ddof=1)],
        [np.mean(shift[:, 1]), np.std(shift[:, 1], ddof=1)]
    ])
