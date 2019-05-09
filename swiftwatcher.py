import swiftwatcher.process_video as pv
import os

# File configs
video_directory = "videos/gdrive_swiftvideos/"
filename = "ch04_20170531210856.mp4"
save_directory = video_directory + os.path.splitext(filename)[0]

# Extraction configs
start = "00:21:30:000"
end = "00:23:30:000"
batch_size = 1000

# Create FrameStack object.
frameStack = pv.FrameStack(video_directory, filename)

# Convert total timestamps into batch timestamps using provided configs
batch_timestamps = frameStack.batch_config(start, end, batch_size)

# Extract frames from video file and save them to a directory.
print("[*] Requested time period: {} to {}.".format(start, end))
print("[*] Batch mode: {} batches of size {} will be extracted.".format(len(batch_timestamps), batch_size))
total_frames = 0
for timestamp_pair in batch_timestamps:
    frameStack.read_frames(start_timestamp=timestamp_pair[0], end_timestamp=timestamp_pair[1])
    total_frames += len(frameStack.stack)

    # TODO: Further preprocessing will obviously be required.
    # Only basic steps had been written because of desire to test frameStack class behavior.
    # frameStack.convert_grayscale()
    frameStack.crop_frames_rect(corners=[(745, 617), (920, 692)])

    frameStack.save_frames(save_directory)
print("[========================================================]")
print("[-] Extraction complete. {} total frames extracted.".format(total_frames))
