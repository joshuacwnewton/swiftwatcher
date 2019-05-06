import swiftwatcher.process_video as pv
import os

video_folder = "videos/gdrive_swiftvideos/"
filename = "ch04_20170531210856.mp4"
video_filepath = video_folder + filename
frame_folder = video_folder + os.path.splitext(filename)[0]

frameStack = pv.FrameStack(video_filepath)
frameStack.read_frames()
frameStack.save_frames(frame_folder)

