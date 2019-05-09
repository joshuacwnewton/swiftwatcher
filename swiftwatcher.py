import swiftwatcher.process_video as pv
import os

# File configs
video_directory = "videos/gdrive_swiftvideos/"
filename = "ch04_20170531210856.mp4"
save_directory = video_directory + os.path.splitext(filename)[0]

# Create FrameStack object.
frameStack = pv.FrameStack(video_directory, filename, 5)

print("[*] Reading frames... (This may take a while!)")
while frameStack.frames_read < frameStack.src_framecount:
    frameStack.read_frame(delay=1)
    frameStack.convert_grayscale(0)
    frameStack.crop_frame(0, corners=[(745, 617), (920, 692)])
    frameStack.resize_frame(0, 1000)
    frameStack.save_frame(0, save_directory)
    if frameStack.frames_read % 1000 == 0:
        print("{}/{} frames successfully processed.".format(frameStack.frames_read, frameStack.src_framecount))
frameStack.stream.release()
print("[========================================================]")
print("[-] Extraction complete. {} total frames extracted.".format(frameStack.frames_read))
