import warnings

from .parameters import *

from numpy.typing import NDArray
from scipy.optimize import leastsq, curve_fit
from scipy.ndimage import maximum_filter
from ellipse import LsqEllipse
from sklearn.cluster import AgglomerativeClustering

"""
Transformation of projection plane to poloidal plane
"""

### transfrom projection plane to poloidal plane

def RANSAC_parabola(u,v,sample_size=10,n=1000,epsilon=0.01):
    #runtime is analyzsed as function of number of iteration (n), total number of data (N), and sample size (s)
    """
    perform RANSAC algorithm to determine horizontal parabola parameters in projection plane

    :param u: values of edge points along u (horizontal) axis
    :param v: values of edge points along v (vertical) axis
    :param sample_size: number of data points to be considered within 1 iteration of RANSAC
    :param n: number of iteration
    :param epsilon: residual threshold to classify whether a point is well fitted with the model

    :return: tuple of horizontal parabola parameter, fit score, points that pass the threshold as 2 arrays best_u, best_v
    """
    warnings.filterwarnings("ignore")

    #function for horizontal parabola
    horizontal_parabola = lambda y,a,b,c: a*y**2 + b*y + c
    #bounds of parameters (a,b,c)

    #perform RANSAC
    best_parameter = None
    best_u = []
    best_v = []
    max_score = 0
    for _ in range(n):
        # Randomly choose "sample size" samples
        random_indices = np.random.choice(len(u), size=sample_size, replace=False) #O(s)

        u_samples = u[random_indices]
        v_samples = v[random_indices]

        # fit model to randomly chosen points
        param, _ = curve_fit(horizontal_parabola, v_samples, u_samples) #O(s)

        # calculate score
        residuals = (u - horizontal_parabola(v, *param))**2
        inliers = residuals <= epsilon
        candidate_score = np.sum(inliers)

        if candidate_score > max_score:
            max_score = candidate_score
            best_u = u[inliers]
            best_v = v[inliers]

    #runtime: O(n*N)

    #fit to data points without noises
    best_parameter, _ = curve_fit(horizontal_parabola, best_v, best_u)
    return best_parameter, max_score, best_u, best_v



def poloidal_transformation(ue: NDArray, ve: NDArray, camera_location: tuple[float,float,float],
                            RANSAC_sample_size = 10, RANSAC_n = 1000, RANSAC_epsilon = 0.000001) -> tuple:
    """
    transform data from projection plane to poloidal plane 

    :param ue: edge point of plasma along u axis
    :param ve: edge point of plasma along v axis
    :param camera_location: location of camera in cartesian coordinate (in meters)

    :return: numpy array of location of the two edges
    """
    # determine parameters of horizontal parabola for transformation
    param, num_pass, u_pass, v_pass = RANSAC_parabola(ue,ve,RANSAC_sample_size,RANSAC_n,RANSAC_epsilon)

    ### perform transformation ###
    def mk_poloidal(ue:NDArray, ve:NDArray, camera_location:tuple[float,float,float]) -> tuple: #O(1)
        """
        Transformation function for 1 edge point

        :param ue: edge point along u dimension
        :param ve: edge point along v dimension
        :param camera_location: location of camera in cartesian coordinate
        :return: (R,Z) edge point on poloidal plane
        """
        
        # calculate phi
        def diff_term(param): #O(1)
            """
            equation of differential term found in phi_poloidal function
            """
            a,b,_ = param
            return 2*a*ve + b

        def phi_poloidal(camera_location: tuple[float,float,float], param: tuple) -> float: #O(1)
            """
            equation of poloidal angle to be solved

            :param camera_location: location of camera in cartesian coordinate (in meters)
            :param param: parameters of fitted parabola
            :return: value of poloidal angle phi
            """
            wc, uc, vc = camera_location

            diff_value = diff_term(param)
            nomi = (ue - uc) - diff_value*(ve-vc)
            return np.arctan(nomi/wc)

        #phi for given point ue and ve
        phi_e = phi_poloidal(camera_location,param)

        # calculate for edge on poloidal plane
        def RZ_edge(phi: float, camera_location: tuple[float,float,float]) -> tuple: #O(1)
            """
            calculate the location of edge on poloidal plane (R, Z) from given edge point in projection plane (u,v)

            :param phi: value of poloidal angle of a given point
            :param camera_location: location of camera in cartesian coordinate (in meters)
            :return: R and Z value transformed from u and v
            """
            wc, uc, vc = camera_location
            # calculate for R
            Re = ue * wc / (wc * np.cos(phi) + (ue - uc) * np.sin(phi))

            # calculate for Z
            Ze = vc + (ve - vc) * (wc - Re * np.sin(phi)) / wc

            return Re, Ze

        return RZ_edge(phi_e,camera_location)

    ## apply transformation functions to all points from RANSAC
    R, Z = [], []
    for u, v in zip(u_pass, v_pass): #O(n)
        Ri, Zi = mk_poloidal(u,v,camera_location)
        R.append(Ri)
        Z.append(Zi)

    return np.array(R),np.array(Z),param, num_pass, u_pass, v_pass

def circle_fit(R:NDArray, Z:NDArray, init_guess:NDArray = (TT1_major_radius,0,TT1_minor_radius)) -> tuple:
    """
    determine parameters of best fit circle to the data on poloidal plane
    :param R: edge data along R dimension
    :param Z: edge data along Z dimension
    :param init_guess: intial guess for (Rc,Zc, minor_radius)
    :return: best fit parameters and covariance matrix
    """
    def circle_eqn(params, x, y):
        xc, yc, r = params
        return (x - xc) ** 2 + (y - yc) ** 2 - r ** 2

    circle_params, circle_var, _, _, _ = leastsq(circle_eqn, x0=init_guess, args=(R, Z), full_output=True)

    return circle_params, circle_var, R, Z

def RANSAC_circle(R,Z,sample_size=10,n=1000,epsilon=0.001,circle_init_guess = (TT1_major_radius,0,TT1_minor_radius)):
    """
    perform RANSAC algorithm to determine circle parameters in proloidal plane

    :param R: values of edge points along R (horizontal) axis
    :param Z: values of edge points along Z (vertical) axis
    :param sample_size: number of data points to be considered within 1 iteration of RANSAC
    :param n: number of iteration
    :param epsilon: residual threshold to classify whether a point is well fitted with the model

    :return: tuple of circle parameter (center, radius), fit score, points that pass the threshold as 2 arrays best_u, best_v 
    """
    #circle equation
    def circle_eqn(params, x, y):
        xc, yc, r = params
        return (x - xc) ** 2 + (y - yc) ** 2 - r ** 2

    #perform RANSAC
    best_parameter = None
    best_R = []
    best_Z = []
    max_score = 0
    for _ in range(n):
        # Randomly choose "sample size" samples
        random_indices = np.random.choice(len(R), size=sample_size, replace=False)

        R_samples = R[random_indices]
        Z_samples = Z[random_indices]

        #fit model to randomly chosen points
        (R0,Z0,r), circle_var, _, _, _ = leastsq(circle_eqn, x0=circle_init_guess, args=(R_samples, Z_samples), full_output=True)
        #calculate score

        #distance of point to center of circle then subtract by radius
        distances = np.abs(np.sqrt((R-R0)**2 + (Z-Z0)**2) - r)
        inliers = distances <= epsilon
        candidate_score = np.sum(inliers)

        #check score
        if candidate_score > max_score:
            max_score = candidate_score
            best_R = R[inliers]
            best_Z = Z[inliers]

    (R0, Z0, r), circle_var, _, _, _ = leastsq(circle_eqn, x0=circle_init_guess, args=(best_R, best_Z),
                                               full_output=True)
    return (R0,Z0,r), circle_var, best_R, best_Z