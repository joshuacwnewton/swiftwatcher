import swiftwatcher.process_video as pv
import os

# File configs
video_directory = "videos/gdrive_swiftvideos/"
filename = "ch04_20170531210856.mp4"
save_directory = video_directory + os.path.splitext(filename)[0]

# Extraction configs
start = "00:22:00:00"
end = "00:32:00:00"
batch_size = 1000

# Create FrameStack object.
frameStack = pv.FrameStack(video_directory, filename)

# Convert timestamp into batch timestamps using configs
batch_timestamps = frameStack.batch_config(start, end, batch_size)

# Extract frames from video file and save them to a directory.
frameStack.read_frames(start_timestamp=start, end_timestamp=end)
frameStack.convert_grayscale()
frameStack.crop_frames_rect(corners=[(745, 617), (920, 692)])
frameStack.save_frames(save_directory)

# TODO: Come up with a plan for handling large video file! (See below for brainstorming?)

# self.size = int((self.end_index - self.start_index) * (desired_fps / self.src_fps))
# if self.size > 10000:
#     print("[!] You have requested more than 1000 frames. ({}) This may take a while.".format(self.size))
#     print("[?] Would you like to continue? [Y/N]")
#     while True:
#         choice = input('--> ')
#         if choice is 'Y' or choice is 'y':
#             pass
#         elif choice is 'N' or choice is 'n':
#             raise Exception("[!] Poor read configuration. Please reconfigure.")
#         else:
#             print("[-] Invalid choice. Please try again.")
#            continue
#         break
