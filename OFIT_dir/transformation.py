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

### simulate projection ###
def simulate_projection(R0: float, r: float, camera_location: tuple[float, float, float], n_samples: int = 1000) -> tuple[NDArray, NDArray]:  # create projection image from simulated shell
    """
    Function to create image on the projection plane given major and minor radius of plasma

    :param R0: plasma major radius in meters
    :param r: plasma minor radius in meters
    :param camera_location: location of camera in cartesian coordinate
    :param n_samples: number of edge points
    :return: 2 numpy arrays of plasma edge in projection plane (u,v)
    """
    wc, uc, vc = camera_location  # extract location of camera

    theta_poloidal = np.linspace(0, 2 * np.pi, n_samples)  # angle on poloidal plane [0,2*pi)

    # plasma edge in tokamak coordinate
    def mk_Re(theta):
        return R0 + r * np.cos(theta)

    def mk_Ze(theta):
        return r * np.sin(theta)

    # equation for solving for toroidal angle of plasma
    def phi_proj(theta):
        Re, Ze = mk_Re(theta), mk_Ze(theta)

        return np.arcsin( (Re - (Ze - vc) * -np.tan(theta))/np.sqrt(uc**2 + wc**2) ) - np.arctan( uc / wc )

    # convert to projection coordinate
    def cal_ue_ve(phi, theta):
        Re, Ze = mk_Re(theta), mk_Ze(theta)

        # calculate for ue in projection plane
        ue = wc * (Re * np.cos(phi) - uc) / (wc - Re * np.sin(phi)) + uc

        # calculate for ve in projection plane
        ve = wc * (Ze - vc) / (wc - Re * np.sin(phi)) + vc
        return ue, ve

    # suppress runtime warning
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="The iteration is not making good progress")

    # calculate phi from each theta
    phi_e = [phi_proj(theta_val) for theta_val in theta_poloidal]

    # convert to projection plane
    projection_coordinate = np.array([cal_ue_ve(phi, theta) for phi, theta in zip(phi_e, theta_poloidal)])
    ue, ve = projection_coordinate[:, 0], projection_coordinate[:, 1]

    return ue, ve

def cluster(u: NDArray, v: NDArray) -> tuple[NDArray, NDArray]:
    """
    cluster given data into two edges (cluster) by AgglomerativeClustering with n_samples = 2

    :param u: edge data along u dimension
    :param v: edge data along v dimension
    :return: clustered data into two edges
    """
    model = AgglomerativeClustering(n_clusters=2)
    Edge = np.column_stack((u, v))
    labels = model.fit_predict(Edge)

    clusters = np.unique(labels)
    clusters_arr = {cluster: Edge[labels == cluster] for cluster in clusters}

    return clusters_arr[0], clusters_arr[1]  # return the two clusters


### transfrom projection plane to poloidal plane

def RANSAC_ellipse(u,v,sample_size=10,n=1000,epsilon=0.000001):
    """
    perform RANSAC algorithm to determine ellipse parameters in projection plane

    :param u: values of edge points along u (horizontal) axis
    :param v: values of edge points along v (vertical) axis
    :param sample_size: number of data points to be considered within 1 iteration of RANSAC
    :param n: number of iteration
    :param epsilon: residual threshold to classify whether a point is well fitted with the model

    :return: tuple of ellipse parameter (center, semi-major, semi-minor, rotation), fit score, points that pass the threshold as 2 arrays best_u, best_v 
    """
    best_parameter = None
    best_u = None
    best_v = None
    max_score = 0
    for _ in range(n):
        # Randomly choose "sample size" samples
        random_indices = np.random.choice(len(u), size=sample_size, replace=False)

        u_samples = u[random_indices]
        v_samples = v[random_indices]

        #fit model to randomly chosen points
        sample_data = np.array(list(zip(u_samples,v_samples)))
        reg = LsqEllipse().fit(sample_data)
        param = reg.as_parameters()

        center, a, b, phi = param
        u0, v0 = center

        u0_rot = u0*np.cos(phi) - v0*np.sin(phi)
        v0_rot = u0*np.sin(phi) + v0*np.cos(phi) 
        #calculate score
        candidate_score = 0

        u_curr_pass = []
        v_curr_pass = []
        for ui, vi in zip(u,v):
            #transform data point to rotated coordinate
            ui_rot = ui*np.cos(phi) - vi*np.sin(phi)
            vi_rot = ui*np.sin(phi) + vi*np.cos(phi)

            if (ui_rot - u0_rot)**2/a**2 < 1:
                residual = min(
                    (b * np.sqrt(1-((ui_rot - u0_rot)/a)**2) + v0_rot - vi_rot)**2,
                    (-b * np.sqrt(1-((ui_rot - u0_rot)/a)**2) + v0_rot - vi_rot)**2
                )

            else: residual = np.inf

            if residual <= epsilon:
                candidate_score += 1
                u_curr_pass.append(ui)
                v_curr_pass.append(vi)

        #check score
        if candidate_score > max_score:
            max_score = candidate_score
            best_parameter = param
            best_u = u_curr_pass
            best_v = v_curr_pass

    return best_parameter, max_score, best_u, best_v

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

def center_shift(R:NDArray, Z:NDArray, init_guess:NDArray = (TT1_major_radius,0,TT1_minor_radius)) -> tuple:
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

    return circle_params, circle_var

def hough_circle(x_arr, y_arr, cx_range = np.linspace(0.45,0.85,90), cy_range=np.linspace(-0.05,0.05,10),bins_number=500,maximum_neighbor_size=10,vote_threshold=100):
    """
    hough transform to detect circles

    :param x_arr: array of data along x axis
    :param y_arr: array of data along y axis
    :param cx_range: all values of cx (circle center along x) to be considered in hough space
    :param cy_range: all values of cy (circle center along y) to be considered in hough space
    :param bins_number: range of neighbor to be considered for peaks in voting
    """
    def all_circle_pass_through_point(x,y):
        # determine allpossible values of rs based on cx and cy and return meshgrid for further voting
        cxs, cys = np.meshgrid(cx_range, cy_range)
        rs = np.sqrt((cxs-x)**2 + (cys-y)**2)
        return cxs, cys, rs
    
    def vote(all_cxs, all_cys, all_rs):
        # count the vote numbers from hough space
        h, (ex, ey, er) = np.histogramdd(np.array([all_cxs, all_cys, all_rs]).T, bins = bins_number)
        return h, ex, ey, er
    
    #extract points in hough space form input data
    all_cxs, all_cys, all_rs = np.array([]),np.array([]),np.array([])
    for point in zip(x_arr, y_arr):
        x,y = point
        cxs, cys, rs = all_circle_pass_through_point(x,y)
        all_cxs = np.append(all_cxs, cxs)
        all_cys = np.append(all_cys, cys)
        all_rs = np.append(all_rs, rs)

    #vote counting
    h, ex, ey, er = vote(all_cxs, all_cys, all_rs)

    #find local maxima peaks of highest vote
    filtered_h = maximum_filter(h, size = maximum_neighbor_size, mode = "constant", cval = 0)
    mask_local_maxima = (h == filtered_h) & (h > vote_threshold)

    #extract higest voting peaks
    local_maxima_index = np.argwhere(mask_local_maxima)
    if len(local_maxima_index) == 0:
        return None, None, None

    max_vote_index = np.argmax(h[mask_local_maxima])
    max_ix,max_iy,max_ir = local_maxima_index[max_vote_index]
    #ex ey er are bin edges average between left and right edge to get center value
    x0 = (ex[max_ix] + ex[max_ix+1])/2
    y0 = (ey[max_iy] + ey[max_iy+1])/2
    r0 = (er[max_ir] + er[max_ir+1])/2

    #return circle parameters
    return x0,y0,r0

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