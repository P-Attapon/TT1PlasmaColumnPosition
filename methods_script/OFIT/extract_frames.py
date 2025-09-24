import cv2
import os

def extract_frames(video_path, output_folder):
    """
    Extracts all frames from a video and saves them as individual image files.

    Args:
        video_path (str): The path to the input video file.
        output_folder (str): The path to the folder where frames will be saved.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder,exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 1
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If 'ret' is False, it means we have reached the end of the video
        if not ret:
            break

        # Construct the filename for the current frame
        frame_filename = os.path.join(output_folder, f"{frame_count}.jpg")

        # Save the frame as an image file
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")

if __name__ == "__main__":
    new_shots = [1641, 1643, 2766]

    for curr_shot in new_shots:
        # Specify the path to your video file
        video_file = f"resources/fullShotData/{curr_shot}/{curr_shot}.avi"

        # Specify the output folder for the frames
        frames_output_dir = f"resources/TTI frame/{curr_shot}/{curr_shot}_frames_jpg"

        # Call the function to extract frames
        extract_frames(video_file, frames_output_dir)

