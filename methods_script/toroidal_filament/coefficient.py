"""
Calculate power series coefficients
"""

from .parameters import coil_angle_dict as angle_dict
from .parameters import all_arrays, base_decimal_precision, probe_lst_to_str, shift_domain
from .plasma_shift import make_taylor
from .DxDz import cal_newton_DxDz as cal_DxDz
from .signal_strength import  cal_signal
from numpy.typing import NDArray
from typing import Callable
import numpy as np
import scipy as sc
from tqdm import tqdm
import pickle
import os

current_dir = os.path.dirname(__file__)

def cal_alpha(DxDz_method: Callable,a:float,taylor_order:int ,horizontal_shift: float, coil_angle: NDArray[np.float64],
            vertical_range: NDArray[np.float64]) -> tuple:
    """
    calculate alpha coefficient from curve fitting

    :param:
    DxDz_method (Callable): method for calculation of Dx & Dz (cal_newton_DxDz/cal_approx_DxDz)
    :param Dz: point to be evaluated in taylor series
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

def cal_beta(DxDz_method: Callable,a: float,taylor_order:int,vertical_shift: float, coil_angle: NDArray[np.float64],
             horizontal_range: NDArray[np.float64] ) -> tuple:
    """
    calculate beta coefficient from curve fitting

    :param DxDz_method (Callable): method for calculation of Dx & Dz (cal_newton_DxDz/cal_approx_DxDz) \n
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


def mk_coefficient_dictionary():
    """
    Create dictionaries of alpha and beta coefficients with valid probe arrays as keys

    :return : alpha_dict, beta_dict
    """

    #create empty dictionaries
    alpha_dict = dict()

    #fill in names of arrays for both coefficients
    for probe_arr in all_arrays:
        arr_str = probe_lst_to_str(probe_arr)
        alpha_dict[arr_str] = None

    beta_dict = alpha_dict.copy()

    return alpha_dict, beta_dict

"""
Nested dictionary implementation
"""

def mk_nested_dict_coeff(shift_domain, taylor_order=3, decimal_precision = base_decimal_precision):
    """
    Create pickle file containing two dictionaries (alpha, beta) of nested dictionaries for each set of magnetic probes
    using rounded precision

    !value of decimal_precision must be same as the one in plasma_shift!

    :param shift_domain: domain of shift_value for both direction
    :param taylor_order: order of taylor series to be fitted
    :param decimal_precision: precision of decimal rounding 
    """
    #round values to have finite decimal points 
    shift_domain = np.round(shift_domain,decimal_precision)

    def mk_shift_dict(lst_angles:list[float], shift_steps:list[float],coeff_function:callable,taylor_order:int):
        """
        calculate coefficients and covariance of shift for one probe array
        Args:
            lst_angles: angles of each probes in radian
            shift_steps: domain of curve to be fit
            coeff_callable: function to calculate coefficients alpha/beta
            taylor_order: order of taylor expansion

        Returns: tree of coefficients
        """
        coeff_dict = dict()
        for shift_value in shift_steps:
            shift_key = str(shift_value)

            shift_coeff_cov = coeff_function(cal_DxDz,0,taylor_order,shift_value,lst_angles,shift_domain)
            
            coeff_dict[shift_key] = shift_coeff_cov
        
        return coeff_dict
    
    #create empty dictionaries with probe keys
    alpha_dict,beta_dict = mk_coefficient_dictionary()
    for probe_arr in tqdm(all_arrays):
        #retrieve installed angle of each probe
        arr_angles = [angle_dict[probe_num] for probe_num in probe_arr]
        #convert list of probe numbers to string to be used as keys in dict
        arr_str = probe_lst_to_str(probe_arr)

        #create trees
        alpha_nested_dict = mk_shift_dict(arr_angles,shift_domain,cal_alpha,taylor_order)
        beta_nested_dict = mk_shift_dict(arr_angles,shift_domain,cal_beta,taylor_order)

        #fill trees into dictionaries
        alpha_dict[arr_str] = alpha_nested_dict
        beta_dict[arr_str] = beta_nested_dict

    output_path = os.path.join(current_dir,"coefficient_nested_dict.pkl")
    with open(output_path,"wb") as nested_dict_file:
        pickle.dump(alpha_dict, nested_dict_file)
        pickle.dump(beta_dict,nested_dict_file)

mk_nested_dict_coeff(np.arange(-shift_domain,shift_domain,0.001)) #all shift values by 1 mm