import swiftwatcher.process_video as pv
import os

# Side-view videos
# videos/gdrive_swiftvideos/ch04_20170531210856.mp4  corners=[(745, 617), (920, 692)]
# videos/gdrive_swiftvideos/ch02_20160513200533.mp4  corners=[(940, 450), (1110, 550)]
# videos/usb_CHSWPRES/ch04_20170518205849.mp4        corners=[( , ), ( , )]

# Top-view videos
# videos/gdrive_swiftvideos/ch01_20160513202121.mp4

# File configs
video_directory = "videos/gdrive_swiftvideos/"
filename = "ch01_20160513202121.mp4"
save_directory = video_directory + os.path.splitext(filename)[0]

# Configs for fetching previously saved frames
reuse = False
frame_index = 24894
stack_size = 400

# Create FrameStack object. (FIFO stack, reading new frame pushes frames through stack)
frameStack = pv.FrameStack(video_directory, filename, stack_size)

# Convert desired_fps into delay between frames (assumes constant framerate)
delay = round(frameStack.src_fps / frameStack.fps) - 1

if not reuse:
    # Code to read frames from video file
    # (Primarily used to make videos easier to parse for interesting frames to test with.
    # Will be returned to when parsing new video files to track birds.)
    print("[========================================================]")
    print("[*] Reading frames... (This may take a while!)")
    while frameStack.frames_read < frameStack.src_framecount:
        # Shift frames through stack, read new frame
        success = frameStack.read_frame_from_video(delay)

        # Process new frame and save
        if success:
            frameStack.save_frame_to_file(save_directory)
        else:
            raise Exception("read_frame() failed before expected end of file.")

        # Status updates
        if frameStack.frames_read % 1000 == 0:
            print("[-] {}/{} frames successfully processed.".format(frameStack.frames_read,
                                                                    frameStack.src_framecount))
    frameStack.stream.release()
    print("[========================================================]")
    print("[-] Extraction complete. {} total frames extracted.".format(frameStack.frames_read))
else:
    # Code to read frames from files
    # Fills frameStack (i.e. reads stack_size number of frames) starting at specified frame_index.
    frameStack.stream.release()  # VideoCapture not needed if frames are being reused

    while frameStack.frames_read < stack_size:
        success = frameStack.load_frame_from_file(save_directory, frame_index)
        frame_index += (1+delay)

        if success:  # Test to see if frames were loaded correctly
            frameStack.segment_frame()
            frameStack.convert_grayscale()
            frameStack.crop_frame(corners=[(745, 617), (920, 692)])  # top-left [w,h], bottom-right [w,h]
            frameStack.resize_frame(1000)
            frameStack.save_frame_to_file(save_directory, folder_name="! test")
