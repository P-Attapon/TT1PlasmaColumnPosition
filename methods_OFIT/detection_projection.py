import cv2
from numpy.typing import NDArray
from .parameters import *

"""
Functions to perform pixel edge detection and transform to Tokamak's projection plane
"""

def set_ROI(image:NDArray, y1:int,x1:int,y2:int,x2:int) -> NDArray:
    """
    extract rectangular ROI from image covering (y1,x1):(y2,x2)

    :param image: image to retreive ROI
    :param y1: row of starting pixel
    :param x1: column of starting pixel
    :param y2: row of last pixel
    :param x2: column of last pixel
    :return: ROI from image
    """

    return image[y1:y2,x1:x2].copy()

def kernel_filter(image:NDArray, kernel: NDArray = np.array([
    [1,2,3,0,-3,-2,-1],
    [1,2,3,0,-3,-2,-1],
    [1,2,3,0,-3,-2,-1]], dtype = np.float32)) -> NDArray:
    """
    apply kernel to filter given image
    :param image: image to be applied kernel to
    :param kernel: kernel applied to image
    :return: kernel applied image
    """
    return cv2.filter2D(image,-1,kernel)

def mk_rotation_matrix(rotation: tuple[float, float]):
    """
    :param rotation: angle between optical axis, transverse plane, and saggital plane respectively
    :return: camera's rotation matrix
    """
    tranverse, saggital = rotation

    # rotational matrix
    #1st rotation around v axis
    alpha = np.deg2rad(tranverse)
    Rv = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])

    #rotation order Rx1 -> Rx2 -> Rx3
    #bring optical axis to plane -> saggital angle -> tranverse angle
    R = Rv

    return R

def mk_intrinsic_matrix(linear_pix_density:tuple[float,float],y0:float,x0:float) -> NDArray:
    """
    create camera's intrinsic matrix from given parameters

    :param y0: principle pixel point along row direction
    :param x0: principle pixel point along column direction
    :param linear_pix_density: linear density of pixel along x and y axis respectively (both are 1/K_scale for square pixels)
    :return: numpy array of camera's intrinsic matrix
    """
    fy,fx = linear_pix_density
    M_int = np.array([
        [fx,0,x0,0],
        [0,fy,y0,0], 
        [0,0,1,0]
    ])

    return M_int

def mk_projection_matrix(rotation: tuple[float,float] = TT1_camera_rotation,translation:tuple[float,float,float] = TT1_camera_translation,
                      principle_point:tuple = TT1_camera_principle_point,linear_pix_density:tuple[float,float] = TT1_linear_pixel_density):
    """
    :param rotation: angle between optical axis, transverse plane, and saggital plane respectively
    :param translation: translation of camera from origin
    :param principle_point: principle pixel point in pixel (y0,x0)
    :param linear_pix_density: linear density of pixel along x and y axis respectively (both are 1/K_scale for square pixels)
    :return: camera's projection matrix
    """
    
    y0, x0 = principle_point

    #create rotation matrix
    R = mk_rotation_matrix(rotation)

    #translation vector
    wc,uc,vc = translation
    t = -R @ np.array([uc,vc,wc])

    # intrinsic matrix
    M_int = mk_intrinsic_matrix(linear_pix_density,y0,x0)

    # extrinsic matrix
    M_ext = np.column_stack((R, t))
    M_ext = np.vstack((M_ext, np.array([0,0,0,1])))

    #projection matrix
    P = M_int @ M_ext

    return P

def max_intensity(row:NDArray):
    """
    :param row: pixel row intensity from image
    :return: the row intensity (the input)    
    """
    return row

def max_gradient(row:NDArray):
    """
    :param row: pixel row intensity from image
    :return: the gradient within row intensity 
    """
    return np.gradient(row)

def find_edge(image,start_row:int, left_ROI,right_ROI,exclusion_set: set,
              n_peaks:int = 1,window_size:int = 0,  detection_method_callable:callable = max_intensity):
    """
    detect plasma edge within given image or ROI

    :param image: input image
    :param start_row: the index of first pixel in ROI (row, column)
    :param n_peaks: number of pixels to be detected within one row
    :param window_size: limit the search to maximum distance from the last detected column
    :param detection_method: cataegory used to find plasma edge, built-in -> (max_intensity & max_gradient)
    :return: array of detected pixel edge
    """
    #Create empty arrays to store the result
    column_result, row_result = [], []

    #intialize window domain for first row as full first row in ROI
    left_window, right_window = left_ROI[0], right_ROI[0]

    row_in_ROI = range(start_row, start_row + len(left_ROI))
    for index, row in enumerate(row_in_ROI):
        ### find peak pixels ###
        row_intensity = detection_method_callable(image[row, left_window:right_window])
        max_intensity_indices = np.argsort(row_intensity)[::-1]

        #extract n pixels with highest intensity ignoring pixels in exclusion_set
        filtered_column_peaks = []

        i = 0
        while (i<len(max_intensity_indices)) & (len(filtered_column_peaks) < n_peaks):
            #convert index in list to actual pixel column
            pixel_column = max_intensity_indices[i] + left_window

            if ((pixel_column, row) not in exclusion_set):
                filtered_column_peaks.append(pixel_column)
            i += 1
        
        # store detected pixels of current row
        column_result.extend(filtered_column_peaks)
        row_result.extend([row]*len(filtered_column_peaks))

        ### update window search domain for next iteration ###

        if window_size == 0 or len(filtered_column_peaks) == 0:
            left_window, right_window = left_ROI[index], right_ROI[index]

        else:
            #update domain of window while keeping domain within ROI
            column_peak = filtered_column_peaks[0] #use pixel with highest intensity as window center

            left_window = max(left_ROI[index], column_peak - window_size)
            right_window = min(right_ROI[index], column_peak + window_size)
    
    return np.array(column_result), np.array(row_result)

def pix_to_projection(x_arr, y_arr,projection_matrix: NDArray, principle_point: tuple[int, int], KB: float=0) -> tuple[NDArray,NDArray]:
    """
    convert edge from pixel coordinate to projection plane

    :param edge: detected edge pixels
    :param projection_matrix: camera's 3x4 projection matrix
    :param principle_point: principle pixel point in pixel (y0,x0)
    :param KB: barrelling factor
    :return: (u,v) points of edge on projection plane
    """

    y0, x0 = principle_point

    #barrelling correction
    barrel_edge = np.array([
        [
            (x - x0) * (1 + KB * np.sqrt( (x-x0)**2 + (y-y0)**2 )) + x0,
            (y - y0) * (1 + KB * np.sqrt( (x-x0)**2 + (y-y0)**2 )) + y0
         ]
        for x,y in zip(x_arr,y_arr)
    ])

    P = projection_matrix

    wc,uc,vc = (0.8310871342332974, 0.5193208789485408, -0.036)
    K = 2.5E-4

    #perform projection
    u_arr,v_arr = np.array([]), np.array([])

    for x,y in barrel_edge:
        # #form matrix to solve for u and v
        # MatA = np.array([
        #     [P[0,0], P[0,1], -x],
        #     [P[1,0], P[1,1], -y],
        #     [P[2,0], P[2,1], -1]
        # ])
        # MatB = np.array([-P[0,3], -P[1,3], -P[2,3]])

        # u, v, _ = np.linalg.solve(MatA, MatB) #solve for u, v, w_tildae

        u = wc * K * (x - x0) + uc
        v = wc * K * (y - y0) + vc

        u_arr = np.append(u_arr, u)
        v_arr = np.append(v_arr, v)

    return u_arr, v_arr
