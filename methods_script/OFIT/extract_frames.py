"""
Extract each frames from video and save to "resources.TTI" frame directory
"""
import os
import cv2

root_dir = os.path.dirname(os.path.abspath(__file__))

def extract_frames_from_video(out_dir, video_path):
    """
    Extracts frames from a video file and saves them as image files.

    Parameters:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.
        every_nth (int): Save every nth frame (default = 1, i.e., all frames).
    """
    #create directory to keep all frames
    os.makedirs(out_dir, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frame_count = 1

    while True:
        ret, frame = cap.read()

        if not ret: break #video ends

        filename = os.path.join(out_dir, str(frame_count) + ".jpg")
        cv2.imwrite(filename,frame)

        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {out_dir}")   

    return

if __name__ == "__main__":
    video_path = input("Enter path to video: ")
    shot_no = input("Enter shot number: ")

    extract_frames_from_video(video_path,shot_no)