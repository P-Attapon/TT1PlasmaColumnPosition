import numpy as np
import scipy as sc
from numpy.typing import NDArray
from .signal_strength import cal_signal
from typing import Callable

from .geometry_TT1 import coil_angle_dict, R0, base_decimal_precision
from .DxDz import cal_newton_DxDz
import pandas as pd
import warnings
import pickle
from tqdm import tqdm
from pathlib import Path


"""
Calculate alpha and beta for curve fitting
"""
# power series for curve fitting
def make_taylor(order,a) -> Callable:
    def taylor_polynomial(x,*coeff):
        return sum([coeff[i] * (x-a)**i for i in range(order + 1)])
    return taylor_polynomial

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

def probe_lst_to_str(lst):
    """
    convert probe numbers stored in list to string for coefficients dictionaries
    """
    arr_str = ""
    for i, probe_num in enumerate(lst):
        if i == 0:
            arr_str += str(probe_num)
        else: arr_str += " " + str(probe_num)
    return arr_str

def coefficient_lookup(val, dict, max_val = 0.13,decimal_precision = base_decimal_precision):
    round_val = round(val,decimal_precision)

    key = str(round_val) if round_val != 0 else "0.0"

    if abs(round_val) > abs(max_val):
        return dict["0.13"] if round_val > 0 else dict["-0.13"]
    return dict[key]

#read coefficient file
toroidalFilament_dir = Path(__file__).resolve().parent
pkl_path = toroidalFilament_dir / "coefficient_nested_dict.pkl"
with open(pkl_path, "rb") as coefficients_file:
    alpha_dict = pickle.load(coefficients_file)
    beta_dict = pickle.load(coefficients_file)
    
def cal_shift(DxDz_method: Callable, taylor_order:int,signal: list[float], est_horizontal_shift: float, est_vertical_shift: float, probe_number: list[int],
              alpha_vertical_range = np.linspace(-0.05,0.05,101), beta_horizontal_range = np.linspace(-0.05,0.05,101)) -> NDArray:
    """
    calculate DeltaX and DeltaZ from power series of Dx & Dz based on estimation of shift value

    :param:
    DxDz_method (Callable): method of calculation for Dx and Dz
    taylor_order (int): order of taylor's polynomial starting from 0
    signal (list[float]): magnetic signal at each magnetic probe in coil_angle
    est_horizontal_shift (float): estimate shift along x/radial direction
    est_vertical_shift (float): estimate shift along z/vertical direction
    probe_number (list[float]): number of used probes GBPXT

    :return:
    matrix: [[x_shift, x_shift_uncertainty],
             [z_shift, z_shift_uncertainty]]
    """
    #look up probe angles of each probes
    coil_angle = [coil_angle_dict[probe] for probe in probe_number]

    #calculate Dx Dz
    Dx, Dz = DxDz_method(signal, coil_angle)

    #convert probe numbers to string keys in dictionaries
    probe_key = probe_lst_to_str(probe_number)

    #look up the coefficients
    alpha, a_cov = coefficient_lookup(est_horizontal_shift,alpha_dict[probe_key]) ##
    beta, b_cov = coefficient_lookup(est_vertical_shift,beta_dict[probe_key]) ##

    #calculate shift based on Dx Dz and coefficients

    vertical_shift = make_taylor(taylor_order, a = 0)(Dz,*alpha)
    horizontal_shift = make_taylor(taylor_order, a = 0)(Dx,*beta)

    def sigma_f(x, popt, pcov):
        """Compute the propagated uncertainty sigma_f(x)."""
        popt, pcov = np.array(popt),np.array(pcov)
        v = np.array([x ** i for i in range(len(popt))])  # Gradient vector (x^0, x^1, x^2, ...)
        var_f = np.dot(v.T, np.dot(pcov, v))  # v^T @ pcov @ v
        return np.sqrt(var_f)

    #calculate uncertainty from covariances
    vertical_shift_uncertainty = sigma_f(Dz,alpha,a_cov)
    horizontal_shift_uncertainty = sigma_f(Dx,beta,b_cov)

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

#shift progression
def toroidal_filament_shift_progression(time_df:pd.DataFrame,signal_df:pd.DataFrame,probe_number:list[list[int]],taylor_order:int = 3,DxDz_method = cal_newton_DxDz):
    """
    use magnetic signal to calculate plasma shift at each time step for each specified array in magnetic probes

    :param time_df: data frame of recorded time of each magnetic signal input
    :param signal_df: data frame of recorded signal 
    :param probe_number: nested list of probe numbers used in each array (counter-clockwise) eg. [[1,4,7,10],[12,3,6,9]]
    :param taylor_order: order of taylor series used in plasma shift calculation.
    :param DxDz_method: calculation method for Dx & Dz use newton method as default
    :return: valid time stamps and shift with respective errror (np.array(valid_time), np.array(R0_arr), np.array(R0_err_arr), np.array(Z0_arr), np.array(Z0_err_arr))
    """
    #determine number of probe arrays used
    num_result = len(probe_number)

    #create blank lists for appendind results of plasma shift
    R0_arr, R0_err_arr = [[0] for _ in range(num_result)], [[0]for _ in range(num_result)]
    Z0_arr,Z0_err_arr = [[0]for _ in range(num_result)], [[0]for _ in range(num_result)]
    valid_time = [[0] for _ in range(num_result)]

    for t, signal in tqdm(zip(
        time_df.to_numpy(),signal_df.to_numpy()
    ),total = len(time_df)):
        #retreive signals for each probe arrays
        signal_df = [[signal[coil] for coil in group] for group in probe_number]

        #calculate shift for each probe arrays
        for i,s in enumerate(signal_df):

            est_R_shift, est_Z_shift = R0_arr[i][-1], Z0_arr[i][-1]
            if abs(est_R_shift) > 0.13:
                if est_R_shift < -0.13: est_R_shift = -0.13
                elif est_R_shift > 0.13: est_R_shift = 0.13

            if abs(est_Z_shift) > 0.13:
                if est_Z_shift < -0.13: est_Z_shift = -0.13
                elif est_Z_shift > 0.13: est_Z_shift = 0.13

            shift = cal_shift(DxDz_method=DxDz_method, taylor_order=taylor_order,signal = s,
                            est_horizontal_shift=est_R_shift, est_vertical_shift=est_Z_shift,probe_number=probe_number[i],
                            alpha_vertical_range=np.linspace(-0.13,0.13,151), beta_horizontal_range=np.linspace(-0.13,0.13,151))

            R_shift, R_err = shift[0]
            Z_shift, Z_err = shift[1]

            R0_arr[i].append(R_shift)
            R0_err_arr[i].append(R_err)
            Z0_arr[i].append(Z_shift)
            Z0_err_arr[i].append(Z_err)

            #time with valid shift values
            valid_time[i].append(t)

    return valid_time, R0_arr, R0_err_arr, Z0_arr, Z0_err_arr