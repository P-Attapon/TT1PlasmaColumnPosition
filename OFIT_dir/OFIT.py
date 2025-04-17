import cv2
import numpy as np
import pickle
from pathlib import Path

from tqdm import tqdm

from .parameters import TT1_linear_pixel_density,TT1_camera_principle_point,TT1_camera_translation,TT1_camera_rotation, TT1_dist_coeffs,TT1_image_kernel, TT1_circular_ROIs
from .detection_projection import mk_intrinsic_matrix, mk_projection_matrix,kernel_filter,max_intensity, max_gradient, find_edge,pix_to_projection
from .transformation import poloidal_transformation, RANSAC_circle
from .local_image import rev_image

"""
calculate plasma shift from CCD image without plotting

Coordinate convention
w points out of page        u points to the right
v points up
"""
TT1_intrinsic_matrix = mk_intrinsic_matrix(linear_pix_density=TT1_linear_pixel_density,y0 = TT1_camera_principle_point[0],x0 = TT1_camera_principle_point[1])
TT1_projection_matrix = mk_projection_matrix(TT1_camera_rotation,TT1_camera_translation,TT1_camera_principle_point,TT1_linear_pixel_density)

def process_image(img:np.ndarray,intrinsic_matrix: np.ndarray = TT1_intrinsic_matrix,
                  distortion_coeff: np.ndarray = TT1_dist_coeffs, kernel:np.ndarray = TT1_image_kernel) -> np.ndarray:
    """
    process RGB image to undistorted, edge enhanced, masked, image

    :param img: RGB image of size 1080x1920
    :param intrinsic_matrix: CCD camera's intrinsic matrix
    :param distortion_coeff: CCD camera's distortion coefficient (distortion due to lens etc. to be passed into cv2.undistort)
    :param kernel: kernel to be used for edge enhancedment
    :return: processed image for edge detection
    """
    #image undistortion
    calibration_matrix = intrinsic_matrix[:3,:3]
    img_undistort = cv2.undistort(img, calibration_matrix,distortion_coeff)

    img_blur = cv2.GaussianBlur(img_undistort,(5,5),0)
    img_gs = cv2.cvtColor(img_blur,cv2.COLOR_RGB2GRAY)

    #enhance edge with kernel
    img_kernel = kernel_filter(img_blur, kernel)
    img_kernel_gs = kernel_filter(img_gs)
    img_kernel_hsv = cv2.cvtColor(img_kernel,cv2.COLOR_RGB2HSV)

    img_result = np.zeros(shape = (1080,1920)) # create blank image for final result

    #apply masks to hsv image
    hue_mask = (img_kernel_hsv[:, :, 0] > 40) & (img_kernel_hsv[:, :, 0] < 140)
    sat_mask = (img_kernel_hsv[:, :, 1] > 200)
    val_mask = (img_kernel_hsv[:, :, 2] > 30)

    combined_mask = hue_mask & sat_mask & val_mask

    # indices of hsv images that passes the mask
    indices = np.where(combined_mask)

    img_result[indices] = img_kernel_gs[indices]

    return img_result

# load excluded pixels (port structure of TT1)

#path to pkl file
OFIT_dir = Path(__file__).resolve().parent
pkl_path = OFIT_dir / "TT1_port_pixel.pkl"

with open(pkl_path,"rb") as structure_edge:
    port_set = pickle.load(structure_edge)
def field_edge_detection(img:np.ndarray, TT1_ROIs:dict = TT1_circular_ROIs, exclusion:set = port_set,
                         n_peaks:int = 1, high_window_size: int = 40, low_window_size:int = 60,detection_method:callable = max_intensity) -> tuple:
    """
    detect plasma edge pixels for high and low field side

    :param img: processed edge enhanced image
    :param TT1_ROIs: ROI dictionary of TT1
    :param exclusion: set of excluded pixels (x,y)
    :param n_peaks: number of peaks to be detected per row
    :param high_window_size: detection window size for high field
    :param low_window_size: detection window size for low field
    :param detection_method: method for plasma edge detection (max_intensity/max_gradient)
    :return: edge pixels in high and low fields ((x_high, y_high), (x_low, y_low))
    """

    x_high, y_high = find_edge(image=img, start_row=TT1_ROIs["high_first_row"],
                               start_ROI=TT1_ROIs["ROI_high_x0"], stop_ROI=TT1_ROIs["ROI_high_xf"],
                               exclusion_set=exclusion, n_peaks=n_peaks, window_size=high_window_size,
                               detection_method_callable=detection_method)
    x_low, y_low = find_edge(image=img, start_row=TT1_ROIs["low_first_row"], start_ROI=TT1_ROIs["ROI_low_x0"],
                             stop_ROI=TT1_ROIs["ROI_low_xf"],
                             exclusion_set=exclusion, n_peaks=n_peaks, window_size=low_window_size,
                             detection_method_callable=detection_method)

    return (x_high, y_high), (x_low, y_low)

def field_transformation(x_edge, y_edge, RANSAC_epsilon,projection_matrix:np.ndarray = TT1_projection_matrix,
                         principle_point:tuple[int,int] = TT1_camera_principle_point,camera_location:tuple[float,float,float] = TT1_camera_translation,
                         RANSAC_sample_size = 4, RANSAC_n = 500):

    """
    convert pixel edge to poloidal plane for each field

    :param x_edge: plasma edge pixel along x dimension
    :param y_edge: plasma edge pixel along y dimension
    :param RANSAC_epsilon: residual value for point to be considered plasma edge
    :param projection_matrix: CCD projection matrix
    :param principle_point: CCD principle point
    :param camera_location: CCD camera translation in world coordinate
    :param RANSAC_sample_size: sample size for one RANSAC iteration
    :param RANSAC_n: number of RANSAC iteration
    :return: poloidal plasma shift (R,Z)
    """

    #transform edge pixels to projection plane
    u_edge, v_edge = pix_to_projection(x_edge, y_edge, projection_matrix, principle_point=principle_point)

    # transform edge from projection plane to poloidal plane
    R_edge, Z_edge, _, _, _, _ = poloidal_transformation(u_edge, v_edge,camera_location=camera_location,
                                                         RANSAC_sample_size=RANSAC_sample_size,RANSAC_n=RANSAC_n,RANSAC_epsilon=RANSAC_epsilon)

    return R_edge, Z_edge

def OFIT(
    #image processing
    img:np.ndarray,shot:int,frame:int,intrinsic_matrix: np.ndarray = TT1_intrinsic_matrix,
    distortion_coeff: np.ndarray = TT1_dist_coeffs, kernel:np.ndarray = TT1_image_kernel,
    #edge detection
    TT1_ROIs:dict = TT1_circular_ROIs, exclusion:set = port_set,
    n_peaks:int = 1, high_window_size: int = 40, low_window_size:int = 60,detection_method:callable = max_intensity,
    #transformation
    RANSAC_high_epsilon:float = 0.00005, RANSAC_low_epsilon:float = 0.00005,projection_matrix:np.ndarray = TT1_projection_matrix,
    principle_point:tuple[int,int] = TT1_camera_principle_point,camera_location:tuple[float,float,float] = TT1_camera_translation,
    RANSAC_sample_size = 4, RANSAC_n = 500,

    #centroid shift
    RANSAC_circle_s = 10, RANSAC_circle_n = 500, RANSAC_circle_epsilon = 0.001
):
    """
    calculate centroid shift from OFIT
    :param img: RGB image of size 1080x1920
    :param shot: experimental shot number
    :param frame: frame number of input image
    :param intrinsic_matrix: CCD camera's intrinsic matrix
    :param distortion_coeff: CCD camera's distortion coefficient (distortion due to lens etc. to be passed into cv2.undistort)
    :param kernel: kernel to be used for edge enhancedment
    :param TT1_ROIs: ROI dictionary of TT1
    :param exclusion: set of excluded pixels (x,y)
    :param n_peaks: number of peaks to be detected per row
    :param high_window_size: detection window size for high field
    :param low_window_size: detection window size for low field
    :param detection_method: method for plasma edge detection (max_intensity/max_gradient)
    :param RANSAC_high_epsilon: residual value for point in high edge to be considered plasma edge
    :param RANSAC_high_epsilon: residual value for point in low edge to be considered plasma edge
    :param projection_matrix: CCD projection matrix
    :param principle_point: CCD principle point
    :param camera_location: CCD camera translation in world coordinate
    :param RANSAC_sample_size: sample size for one RANSAC iteration
    :param RANSAC_n: number of RANSAC iteration
    :return: centroid shift in poloidal plane ((R0,Z0,r), cov)
    """
    # check image brightness to eliminate unanalyzable images
    img_brightness = np.mean(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
    if img_brightness < 70 or img_brightness > 130: 
        print(f"image shot {shot} frame {frame} is too dim or too bright returning None")
        return (None, None, None), None

    #process image
    img_processed = process_image(img = img,intrinsic_matrix=intrinsic_matrix,distortion_coeff=distortion_coeff,kernel=kernel)
    #detect edge pixels
    (x_high,y_high), (x_low,y_low) = field_edge_detection(img=img_processed,TT1_ROIs=TT1_ROIs,exclusion=exclusion,
                                                          n_peaks=n_peaks,high_window_size=high_window_size,low_window_size=low_window_size,
                                                          detection_method=detection_method)

    #convert each field to poloidal plane
    R_high, Z_high = field_transformation(x_high,y_high,RANSAC_epsilon=RANSAC_high_epsilon,projection_matrix=projection_matrix,
                                          principle_point=principle_point,camera_location=camera_location,RANSAC_sample_size=RANSAC_sample_size,RANSAC_n=RANSAC_n)
    R_low, Z_low = field_transformation(x_low,y_low, RANSAC_epsilon=RANSAC_low_epsilon,projection_matrix=projection_matrix,
                                        principle_point=principle_point,camera_location=camera_location,RANSAC_sample_size=RANSAC_sample_size,RANSAC_n=RANSAC_n)

    #combine edges and calculate centroid shift
    R, Z = np.append(R_high,R_low), np.append(Z_high,Z_low)
    #initial guess = (TT1_major_radius,0,TT1_minor_radius)
    (R0,Z0,r), cov, _, _ = RANSAC_circle(R,Z,sample_size=RANSAC_circle_s,n=RANSAC_circle_n,epsilon = RANSAC_circle_epsilon)

    return (R0,Z0,r), cov