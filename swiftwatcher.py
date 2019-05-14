import swiftwatcher.process_video as pv
import os

# File configs
# videos/gdrive_swiftvideos/ch04_20170531210856.mp4 corners=[(745, 617), (920, 692)]
# videos/gdrive_swiftvideos/ch02_20160513200533.mp4 corners=[(940, 450), (1110, 550)]
reuse = True
video_directory = "videos/gdrive_swiftvideos/"
filename = "ch04_20170531210856.mp4"
save_directory = video_directory + os.path.splitext(filename)[0]
stack_size = 37

# Create FrameStack object. (FIFO stack, reading new frame pushes frames through stack)
frameStack = pv.FrameStack(video_directory, filename, stack_size)

# Convert desired_fps into delay between frames (assumes constant framerate)
delay = round(frameStack.src_fps / frameStack.fps) - 1

if reuse:
    # Code to read frames from folder
    frame_index = 24047
    while frameStack.frames_read < stack_size:
        success = frameStack.load_frame(save_directory, frame_index)
        frame_index += (1+delay)

        if success:  # Test to see if frames were loaded correctly
            frameStack.save_frame(save_directory, folder_name="extraction-test")
else:
    # Code to read frames from video file
    print("[========================================================]")
    print("[*] Reading frames... (This may take a while!)")
    while frameStack.frames_read < 1080: # frameStack.src_framecount:
        # Shift frames through stack, read new frame
        success = frameStack.read_frame(delay)

        # Process new frame and save
        if success:
            frameStack.convert_grayscale()
            frameStack.crop_frame(corners=[(745, 617), (920, 692)])  # top-left [w,h], bottom-right [w,h]
            frameStack.resize_frame(1000)
            frameStack.save_frame(save_directory)

        # Status updates
        if frameStack.frames_read % 1000 == 0:
            print("[-] {}/{} frames successfully processed.".format(frameStack.frames_read,
                                                                    frameStack.src_framecount))
    frameStack.stream.release()

print("[========================================================]")
print("[-] Extraction complete. {} total frames extracted.".format(frameStack.frames_read))
