import cv2
import os
from glob import glob


# color from shot 961 frame 184
def rev_image(shot_no, frame):
    im_paths = glob(os.path.join(os.getcwd(), "resources", "TTI frame",
                                 f"{shot_no}", f"{shot_no}_frames_jpg", "*.jpg"))

    shot = {path.split('\\')[-1].split('.')[0]: path for path in im_paths}
    shot = cv2.imread(shot[f"{frame}"])
    shot = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
    return shot