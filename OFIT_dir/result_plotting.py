import cv2
import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from matplotlib import patches
import matplotlib
import pickle

from .parameters import TT1_linear_pixel_density,TT1_camera_principle_point,TT1_camera_translation,TT1_camera_rotation, TT1_dist_coeffs,TT1_image_kernel, TT1_circular_ROIs
from .detection_projection import mk_intrinsic_matrix, mk_projection_matrix,kernel_filter,max_intensity, max_gradient, find_edge,pix_to_projection
from .transformation import poloidal_transformation, RANSAC_circle
from .local_image import rev_image

"""
Coordinate convention
w points out of page
u points to the right
v points up
"""

#projection matrix
TT1_intrinsic_matrix = mk_intrinsic_matrix(linear_pix_density=TT1_linear_pixel_density,y0 = TT1_camera_principle_point[0],x0 = TT1_camera_principle_point[1])
TT1_projection_matrix = mk_projection_matrix(TT1_camera_rotation,TT1_camera_translation,TT1_camera_principle_point,TT1_linear_pixel_density)

##TT1 circular ROIs is defined in parameters.py
with open("TT1_port_pixel.pkl","rb") as structure_edge:
    port_set = pickle.load(structure_edge)

def OFIT_plotting(image: NDArray,
         #image correction
         camera_location:tuple[float,float,float] = TT1_camera_translation, intrinsic_matrix: NDArray = TT1_intrinsic_matrix, distortion_coeff: NDArray = TT1_dist_coeffs,
         #kernel blur
         kernel:NDArray = TT1_image_kernel,
         #edge detection
         TT1_ROIs = TT1_circular_ROIs,
         exclusion:set = port_set,n_peaks:int = 1, high_window_size: int = 0, low_window_size:int = 0,detection_method:callable = max_intensity,
         #projection matrix
         projection_matrix:NDArray = TT1_projection_matrix, principle_point:tuple[int,int] = TT1_camera_principle_point,
         #poloical transformation
         RANSAC_sample_size:int = 40, RANSAC_n:int = 1000, RANSAC_high_epsilon = 0.00005,RANSAC_low_epsilon = 0.00005):
    """
    detect plasma edge within given ROI and transform to poloidal plane

    :param image: RGB image from CCD camera
    :param camera_location: location of the CCD camera on the cartesian tokamak coordinate (x,y,z)
    :param intrinsic_matrix: CCD camera's intrinsic matrix
    :param distortion_coeff: CCD camera's distortion coefficient (distortion due to lens etc. to be passed into cv2.undistort)
    :param ROIs_location: collection of first and last pixel of each ROI within the image as [ #first ROI [row_0,column_0,row_1,column_1], #second ROI[row_2,column_2,row_3,column_3]]
    :param kernel: kernel to be used to sharpen edge and/or blur before edge detection
    :param TT1_ROIs: ROIs on image specifying min and max pixel for each row (not necessary circular)
    :param n_peaks: number of pixels to be detected within one row
    :param window_size: limit the search to maximum distance from the last detected column
    :param detection_method: cataegory used to find plasma edge, built-in -> (max_intensity & max_gradient)
    :param projection_matrix: CCD camera's projection matrix
    :param principle_point: camera's pixel principle point (y,x)
    :return: 2 arrays of plasma (unordered high & low filed side)edge on poloidal plane
    """

    #undistort image and convert to grayscale
    calibration_matrix = intrinsic_matrix[:3,:3] #extract calibration matrix from intrinsic matrix
    image = cv2.undistort(image,calibration_matrix,distortion_coeff)

    #blur image
    img_blur = cv2.GaussianBlur(image,(5,5),0)
    img_gs = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)

    #enhance edge by kernel
    image_kernel = kernel_filter(img_blur, kernel)

    #segment image by color
    kernel_hsv = cv2.cvtColor(image_kernel, cv2.COLOR_RGB2HSV)
    kernel_gs = kernel_filter(img_gs)
    #create blank grayscale image
    img_canvas = np.zeros(shape=(1080,1920))

    hue_mask = (kernel_hsv[:,:,0] > 40) & (kernel_hsv[:,:,0] < 140)
    sat_mask = (kernel_hsv[:,:,1] > 200)
    val_mask = (kernel_hsv[:,:,2] > 30)

    combined_mask = hue_mask & sat_mask & val_mask

    #indices of hsv images that passes the mask
    indices = np.where(combined_mask)

    #extract grayscale parts from indices
    img_canvas[indices] = kernel_gs[indices]

    #detect edge within each ROI by provided detection method
    x_high, y_high = find_edge(image=img_canvas, start_row = TT1_ROIs["high_first_row"],start_ROI=TT1_ROIs["ROI_high_x0"],stop_ROI=TT1_ROIs["ROI_high_xf"],
                          exclusion_set = exclusion, n_peaks=n_peaks, window_size=high_window_size,detection_method_callable=detection_method)
    x_low, y_low = find_edge(image=img_canvas,start_row=TT1_ROIs["low_first_row"],start_ROI=TT1_ROIs["ROI_low_x0"],stop_ROI=TT1_ROIs["ROI_low_xf"],
                         exclusion_set = exclusion, n_peaks=n_peaks, window_size=low_window_size,detection_method_callable=detection_method)

    #combine all edge for yield
    x_ret = np.append(x_high,x_low)
    y_ret = np.append(y_high,y_low)
    yield x_ret, y_ret

    #transform pixel to projection plane using projection matrix
    u_high, v_high = pix_to_projection(x_high, y_high, projection_matrix,principle_point=principle_point)
    u_low, v_low = pix_to_projection(x_low, y_low, projection_matrix,principle_point=principle_point)

    yield (u_high,v_high), (u_low,v_low)


    #transform edge from projection plane to poloidal plane
    R_high, Z_high,param_high, pass_high, u_high_pass, v_high_pass = poloidal_transformation(u_high,v_high,camera_location=camera_location,RANSAC_sample_size=RANSAC_sample_size, RANSAC_n=RANSAC_n, RANSAC_epsilon=RANSAC_high_epsilon)
    R_low, Z_low,param_low, pass_low, u_low_pass, v_low_pass = poloidal_transformation(u_low,v_low,camera_location=camera_location,RANSAC_sample_size=RANSAC_sample_size, RANSAC_n=RANSAC_n, RANSAC_epsilon=RANSAC_low_epsilon)

    #combine result into one array
    R_com, Z_com = np.append(R_high,R_low), np.append(Z_high,Z_low)

    yield R_com, Z_com, (param_high, pass_high,u_high_pass, v_high_pass), (param_low, pass_low, u_low_pass,v_low_pass)


path = r"C:\Users\pitit\Documents\02_MUIC_programming\ICPY_441_Senior_project_in_physics\OFIT\resources\OFIT_result\shot_966\\"

kernel = np.array([                                         # kernel to sharpen edge
    [2,3,0,-3,-2],
    [2,3,0,-3,-2],
    [2,3,0,-3,-2]], dtype = np.float32)

def ROI_overlay(image,column,row):
    #function to overlay ROI onto image
    img_height, img_width, _ = image.shape

    for col, ro in zip(column, row):
        if 0 <= ro < img_height and 0 <= col < img_width:
            image[ro-3:ro+3, col-3:col+3] = [0,255,0]  # Green color

from tqdm import tqdm
#163 up to 206
shot_no = 966
# for frame in tqdm(range(146,206)):

lst = list(range(146,351)) # 207
for frame in tqdm(lst):
    shot = rev_image(shot_no,frame)

    result  = []

    # pixel, projection, poloidal
    for i in OFIT_plotting(shot,detection_method=max_intensity,high_window_size=40,low_window_size=60, RANSAC_sample_size=4,RANSAC_n=500,RANSAC_low_epsilon = 0.00005,RANSAC_high_epsilon=0.00005):
        result.append(i)

    x,y = result[0]
    (u1,v1),(u2,v2) = result[1]
    R, Z, (param1, pass_1, u1_pass, v1_pass), (param2, pass_2, u2_pass, v2_pass) = result[2]

    (a1,b1,c1) = param1
    (a2,b2,c2) = param2

    # mark edges pixels on to image
    for xi, yi in zip(x,y):
        shot[yi-3:yi+3,xi-3:xi+3] = [0,0,255]

    ROI_overlay(shot,TT1_circular_ROIs["ROI_high_x0"],TT1_circular_ROIs["high_row_range"])
    ROI_overlay(shot,TT1_circular_ROIs["ROI_high_xf"],TT1_circular_ROIs["high_row_range"])

    ROI_overlay(shot,TT1_circular_ROIs["ROI_low_x0"],TT1_circular_ROIs["low_row_range"])
    ROI_overlay(shot,TT1_circular_ROIs["ROI_low_xf"],TT1_circular_ROIs["low_row_range"])


    (x0,y0,r), cov, best_R, best_Z = RANSAC_circle(R,Z,epsilon=0.007,sample_size = 3, n = 500)
    circle = matplotlib.patches.Circle((x0,y0),radius = r, fill = False)

    fig, (ax0,ax1,ax2) = plt.subplots(3,1, figsize = (8,18))

    ax0.imshow(shot)
    ax0.set_title(f"plasma image, shot: {shot_no}, frame: {frame}")
    ax0.set_xlabel("x [pix]")
    ax0.set_ylabel("y [pix]")
    ax0.grid()

    horizontal_parabola = lambda y,a,b,c: a*y**2 + b*y + c
    y1s = np.linspace(np.min(v1), np.max(v1), 1000)
    x1s = horizontal_parabola(y1s, *param1)
    y2s = np.linspace(np.min(v2),np.max(v2), 1000)
    x2s = horizontal_parabola(y2s,*param2)

    ax1.plot(u1,v1,".", color = "aqua", alpha = 0.8)
    ax1.plot(u1_pass,v1_pass,".", color = "blue")
    ax1.plot(u2,v2,".", color = "pink", alpha = 0.8)
    ax1.plot(u2_pass, v2_pass, ".", color = "red")
    ax1.set_title("projection plane")
    ax1.set_xlabel("u [m]")
    ax1.set_ylabel("v [m]")
    ax1.set_aspect("equal")
    ax1.set_xlim(0.4,0.8)
    ax1.set_ylim(-0.2,0.2)
    ax1.plot(x1s,y1s)
    ax1.plot(x2s,y2s)
    ax1.grid()


    param_round = tuple(round(p,2) for p in (x0,y0,r))
    cov = np.diag(cov)
    sd_round = tuple(round(np.sqrt(c),1) for c in cov)
    ax2.plot(R,Z,".",color = "blue",label = "noise")
    ax2.plot(best_R,best_Z,".",color = "red", label = "real edge")
    ax2.set_title(f"poloidal plane \n (R,Z,r) = {param_round}, s.d. = {sd_round}")#  \n +- {cov_round}")
    ax2.set_xlabel("R [m]")
    ax2.set_ylabel("Z [m]")
    ax2.add_patch(circle)
    ax2.set_aspect("equal")
    ax2.set_xlim(0.4,0.8)
    ax2.set_ylim(-0.2,0.2)
    ax2.legend()
    ax2.grid()

    circle = plt.Circle((x0,y0), r, fill = False)
    ax2.add_patch(circle)


    fig.savefig(path + f"{frame}")
    plt.tight_layout()
    plt.close()
    print(f"{frame} done")