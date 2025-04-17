import cv2
import os
from glob import glob
from pathlib import Path


# color from shot 961 frame 184
def rev_image(shot_no, frame):
    """
    retreive image from default repository structure

    :param shot_no: experimental shot number 
    :param frame: frame number of given shot
    :return: RGB image from given shot_no and frame
    """

    root_dir = Path(__file__).resolve().parent.parent

    #path to image folder
    im_dir = root_dir / "resources" / "TTI frame" / str(shot_no) / f"{shot_no}_frames_jpg"

    # Get all jpg images paths in the folder
    im_paths = glob(str(im_dir /"*.jpg"))

    # Create a dict with frame numbers as keys
    shot = {Path(path).stem: path for path in im_paths}

    #Read image and convert to RGB
    img = cv2.imread(shot[f"{frame}"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def get_frames_for_shot(shot_no:int) -> list[int]:
    """
    determine frames in each shot

    :param shot_no: experimental shot number
    :return: list of all frame numbers
    """
    root_dir = Path(__file__).resolve().parent.parent  # project root
    frame_dir = root_dir / "resources" / "TTI frame" / str(shot_no) / f"{shot_no}_frames_jpg"
    
    if not frame_dir.exists():
        raise FileNotFoundError(f"No such directory: {frame_dir}")
    
    # List all .jpg files and extract frame numbers (stems)
    frame_paths = list(frame_dir.glob("*.jpg"))
    frames = sorted([int(p.stem) for p in frame_paths if p.stem.isdigit()])
    
    return frames