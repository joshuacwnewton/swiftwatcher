import swiftwatcher.process_video as pv
import os

reuse = True
video_folder = "videos/gdrive_swiftvideos/"
filename = "ch04_20170531210856.mp4"
video_filepath = video_folder + filename
save_directory = video_folder + os.path.splitext(filename)[0]

# Create FrameStack object.
frameStack = pv.FrameStack(video_filepath)

if reuse:
    # Reload frames from the save directory of a previous extraction.
    frameStack.load_frames(save_directory)
    frameStack.load_frames_info(save_directory)
else:
    # Extract frames from video file and save them to a directory.
    frameStack.read_frames(desired_fps=(1/30))
    frameStack.save_frames(save_directory)
    frameStack.save_frames_info(save_directory)
