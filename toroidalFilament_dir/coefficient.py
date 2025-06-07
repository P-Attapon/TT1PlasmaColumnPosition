"""
Calculate power series coefficients
"""

from .geometry_TT1 import coil_angle_dict as angle_dict, all_arrays, base_decimal_precision
from .plasma_shift import cal_alpha, cal_beta, probe_lst_to_str
from .DxDz import cal_newton_DxDz as cal_DxDz
from bintrees import AVLTree
import numpy as np
from tqdm import tqdm
import pickle
import os

current_dir = os.path.dirname(__file__)

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
AVLtree implementation 
"""

def mk_AVLtrees_coefficients(shift_domain,taylor_order=3):
    """
    Create pickle file containing two dictionaries (alpha, beta) of AVLtrees for each set of magnetic probes

    :param shift_domain: domain of shift_value for both direction
    :param taylor_order: order of taylor series to be fitted
    """

    def mk_shift_tree(lst_angles:list[float],shift_steps:list[float],coeff_function:callable,taylor_order:int):
        """
        calculate coefficients and covariance of shift for one probe array
        Args:
            lst_angles: angles of each probes in radian
            shift_steps: array containing shift values
            coeff_function: function to calculate coefficients
            taylor_order: order of taylor expansion

        Returns: tree of coefficients
        """
        coeff_tree = AVLTree()
        for shift_amount in shift_steps:
            shift_coeff_cov = coeff_function(cal_DxDz,0,taylor_order,shift_amount,lst_angles,shift_steps)
            coeff_tree.insert(shift_amount,shift_coeff_cov)
        
        return coeff_tree
    
    #create empty dictionaries with probe keys
    alpha_dict, beta_dict = mk_coefficient_dictionary()
    for probe_arr in tqdm(all_arrays):
        #retrieve installed angle of each probe
        arr_angles = [angle_dict[probe_num] for probe_num in probe_arr]
        #convert list of probe numbers to string to be used as keys in dict
        arr_str = probe_lst_to_str(probe_arr)

        #create trees
        alpha_tree = mk_shift_tree(arr_angles,shift_domain,cal_alpha,taylor_order)
        beta_tree = mk_shift_tree(arr_angles,shift_domain,cal_alpha,taylor_order)

        #fill trees into dictionaries
        alpha_dict[arr_str] = alpha_tree
        beta_dict[arr_str] = beta_tree

    output_path = os.path.join(current_dir,"coefficient_tree.pkl")

    with open(output_path,"wb") as coefficient_file:
        pickle.dump(alpha_dict, coefficient_file)
        pickle.dump(beta_dict, coefficient_file)

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

mk_nested_dict_coeff(np.arange(-0.13,0.131,0.001)) #all shift values by 1 mm