import numpy as np
"""
Coordinate convention

w points out of page
u points to the right
v points up
"""

# parameters of CCD camera
TT1_camera_rotation = (0,0)                                # angle betweeen optical axis and transverse plane, angle between optical axis and saggital plane
TT1_camera_translation = (0.8310871342332974, 0.5193208789485408, -0.036)                    # translation of cameara (w,u,v)
TT1_camera_principle_point = (1080//2, 1920//2)             # principle point on camera sensor (vertical, horizontal)
TT1_Kscale = 2.5E-4                                         # scaling factor
TT1_linear_pixel_density = (1/TT1_Kscale, 1/TT1_Kscale)     # linear pixel density along vertical and horizontal direction (fy, fx)
TT1_major_radius = 0.65 #m
TT1_minor_radius = 0.20 #m

TV_distort = 5.9E-3
TT1_dist_coeffs = np.array([TV_distort, 0, 0, 0, 0], dtype = np.float32) #distortion coefficients of camera to be passed into cv2.undistort

TT1_image_kernel = np.array([                                         # kernel to blur image and sharpen edge
    [1,2,3,0,-3,-2,-1],
    [1,2,3,0,-3,-2,-1],
    [1,2,3,0,-3,-2,-1]], dtype = np.float32)

### functions to create circular ROIs ###
def circular_ROI(row_range, center, radius, scale, is_high_field):
    """
    find x value based on row
    """
    def circle_x(y,center, radius,negative_root):
        """
        calculate circle edge based on row number and circle parameters
        """
        #if negative root -> find root with negative sign
        factor = -1 if negative_root else 1
        y0,x0 = center

        if (y-y0)**2 > radius**2: return np.nan

        return factor * np.sqrt(radius**2 - (y-y0)**2) + x0
    #list to store result
    seg_in, seg_out = [], []
    #list for rows of result

    for i in row_range:
        #find x value of outer and inner circle
        x_out = circle_x(i, center,radius,is_high_field)        #outer circle
        x_in = circle_x(i,center,scale*radius, is_high_field)   #inner circle

        #if the value is nan then set to be equal to circle center
        if np.isnan(x_out): x_out = center[1]
        if np.isnan(x_in): x_in = center[1]

        #append result
        seg_in.append(int(x_in))
        seg_out.append(int(x_out))

    return seg_in,seg_out

""" 
parameters of circular ROI

high: high field side
low: low field side

row_range: rows of pixels to be considered
first_row: the first row of pixel where ROI is defined

center: (row,column)center of circular ROI
radius: radius of circular ROI
scale: scaling factor multiplied to radius to make inner circle
"""

#high field parameters
high_row_range = list(range(350,680))
high_first_row = 350 #300
high_center = (560,1280)
high_radius = 500
high_scale = 0.60

#low field parameters
low_row_range = list(range(250,850))
low_first_row = 250
low_center = (540,1085)
low_radius = 500
low_scale = 0.35

#create circular ROIs
#swap order of x0 and xf because in high field start from outer circle to inner
#low field start from inner circle to outer

ROI_high_xf, ROI_high_x0 = circular_ROI(high_row_range,high_center,
                                        high_radius,high_scale,is_high_field=True,)

ROI_low_x0,ROI_low_xf = circular_ROI(low_row_range,low_center,
                                    low_radius,low_scale,is_high_field=False)


TT1_circular_ROIs = {
    "ROI_high_x0":ROI_high_x0, "ROI_high_xf":ROI_high_xf,"high_first_row":high_first_row, "high_row_range":high_row_range,
    "ROI_low_x0":ROI_low_x0, "ROI_low_xf":ROI_low_xf,"low_first_row":low_first_row,"low_row_range":low_row_range
}