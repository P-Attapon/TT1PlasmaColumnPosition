import cv2
import numpy as np
import pickle
from pathlib import Path
import os
from tqdm import tqdm

from .parameters import TT1_linear_pixel_density,TT1_camera_principle_point,TT1_camera_translation,TT1_camera_rotation, TT1_dist_coeffs,TT1_image_kernel, TT1_circular_ROIs
from .parameters import TT1_major_radius as R0
from .detection_projection import mk_intrinsic_matrix, mk_projection_matrix,kernel_filter,max_intensity, max_gradient, find_edge,pix_to_projection
from .transformation import poloidal_transformation, RANSAC_circle, circle_fit
from .extract_frames import extract_frames_from_video

import matplotlib.image as mpimg
import pandas as pd
"""
calculate plasma shift from CCD image without plotting

Coordinate convention
w points out of page        u points to the right
v points up
"""
TT1_intrinsic_matrix = mk_intrinsic_matrix(linear_pix_density=TT1_linear_pixel_density,y0 = TT1_camera_principle_point[0],x0 = TT1_camera_principle_point[1])
TT1_projection_matrix = mk_projection_matrix(TT1_camera_rotation,TT1_camera_translation,TT1_camera_principle_point,TT1_linear_pixel_density)

# load excluded pixels (port structure of TT1)

#path to pkl file
OFIT_dir = Path(__file__).resolve().parent
pkl_path = OFIT_dir / "TT1_port_pixel.pkl"

with open(pkl_path,"rb") as structure_edge:
    port_set = pickle.load(structure_edge)

def process_image(img:np.ndarray,intrinsic_matrix: np.ndarray = TT1_intrinsic_matrix,
                  distortion_coeff: np.ndarray = TT1_dist_coeffs, kernel:np.ndarray = TT1_image_kernel,apply_hsv_mask=True) -> np.ndarray:
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

    #apply masks to hsv image
    if apply_hsv_mask:
        img_kernel_hsv = cv2.cvtColor(img_kernel,cv2.COLOR_RGB2HSV)

        img_result = np.zeros(shape = (1080,1920)) # create blank image for final result

        hue_mask = (img_kernel_hsv[:, :, 0] > 40) & (img_kernel_hsv[:, :, 0] < 140)
        sat_mask = (img_kernel_hsv[:, :, 1] > 200)
        val_mask = (img_kernel_hsv[:, :, 2] > 30)

        combined_mask = hue_mask & sat_mask & val_mask

        # indices of hsv images that passes the mask
        indices = np.where(combined_mask)

        img_result[indices] = img_kernel_gs[indices]
    
    else: img_result = img_kernel_gs

    return img_result

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
                               left_ROI=TT1_ROIs["ROI_high_x0"], right_ROI=TT1_ROIs["ROI_high_xf"],
                               exclusion_set=exclusion, n_peaks=n_peaks, window_size=high_window_size,
                               detection_method_callable=detection_method)
    x_low, y_low = find_edge(image=img, start_row=TT1_ROIs["low_first_row"], left_ROI=TT1_ROIs["ROI_low_x0"],
                             right_ROI=TT1_ROIs["ROI_low_xf"],
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
    (R0,Z0,r), cov, _, _ = circle_fit(R,Z)

    return (R0,Z0,r), cov

def calibration_plane_shift(data_directory,shot_no,frame_step,discharge_begin,discharge_end,edge_detection_image = False):
    """
    Calculate plasma column position shift using transformation from calibration plane
    """

    #function to convert frame number to time with given formula
    frame_to_time = lambda frame: frame/2 + 260

    #function to transform pixel to calibration plane
    pixel_to_calibration = lambda q,edge_pixel, pixel_plane_ratio=0.9: (q - edge_pixel)*pixel_plane_ratio/1000

    calibration_plane_rows = []
    #path of every images in current shot
    img_dir = os.path.join(data_directory,"imgs")

            # Extract frames from video if folder does not exist
    if not os.path.exists(img_dir) or not os.path.isdir(img_dir):
        video_path = os.path.join(data_directory, f"{shot_no}.avi")
        extract_frames_from_video(img_dir, video_path)

    # Get sorted list of images by frame number
    shot_img_paths = sorted(os.listdir(img_dir), key=lambda x: int(Path(x).stem))

    # ---- Process each frame ----
    for frame_no, img_path in tqdm(enumerate(shot_img_paths, start=1),
                                total=len(shot_img_paths),
                                desc="calibration plane"):
        
        # Skip frames according to frame_step
        if frame_no % frame_step != 0:
            continue

        # Calculate calibration plane time
        calibration_plane_time = frame_to_time(frame_no)
        if calibration_plane_time < discharge_begin:
            continue
        if calibration_plane_time > discharge_end:
            break

        # Load image
        img = mpimg.imread(os.path.join(img_dir, img_path))

        # Convert float images to 0-255 uint8
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)

        # Calculate image brightness
        img_brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        if img_brightness < 70 or img_brightness > 130:
            continue

        # Process image
        processed_image = process_image(img, apply_hsv_mask=True)

        # Detect plasma edges
        (x_high, y_high), (x_low, y_low) = field_edge_detection(processed_image)

        # Optional: save edge detection image
        if edge_detection_image:
            x_com, y_com = np.append(x_high, x_low), np.append(y_high, y_low)
            img_detection = img.copy()
            for x, y in zip(x_com, y_com):
                img_detection[y-3:y+3, x-3:x+3] = [255, 0, 0]
            output_dir = Path(os.path.join("result_plot", "edge_detection", str(shot_no)))
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = os.path.join(output_dir, f"{frame_no}.jpg")
            mpimg.imsave(filename, img_detection)

        # Transform y = 0 to center of image
        y_high, y_low = y_high - 1080 // 2, y_low - 1080 // 2

        # Convert to calibration plane
        u_high, v_high = pixel_to_calibration(x_high, 500), pixel_to_calibration(y_high, 0)
        u_low, v_low = pixel_to_calibration(x_low, 500), pixel_to_calibration(y_low, 0)

        # Fit circle using RANSAC
        (u0, v0, radius), circle_var, *_ = RANSAC_circle(np.append(u_high, u_low), np.append(v_high, v_low), epsilon=0.001)

        # Calculate error bars
        all_u = np.append(u_high, u_low)
        all_v = np.append(v_high, v_low)
        residuals = np.sqrt((all_u - u0) ** 2 + (all_v - v0) ** 2) - radius

        dof = len(residuals) - 3  # degrees of freedom
        s_sq = np.sum(residuals ** 2) / dof
        cov_scaled = circle_var * s_sq
        sigma_u0, sigma_v0, sigma_radius = np.sqrt(np.diag(cov_scaled))

        # Append results
        calibration_plane_rows.append([calibration_plane_time, u0 - R0, sigma_u0, v0, sigma_v0, radius, sigma_radius])

    print("Terminated: discharge complete")
    # ---- Create DataFrame ----
    calibration_plane_df = pd.DataFrame(
        calibration_plane_rows,
        columns=["time", "x0", "x0 err", "y0", "y0 err", "r", "r err"]
    )

    return calibration_plane_df