import os
import cv2
import numpy as np
from time import sleep


def ms_to_timestamp(total_ms):
    """Helper function to convert millisecond value into formatted timestamp."""
    # cv2's VideoCapture class provides the position of the video in milliseconds
    # (cv2.CAP_PROP_POS_MSEC). However, as it is easier to think of video in terms of
    # hours, minutes, and seconds, it's helpful to convert to a formatted timestamp.
    total_s = int(total_ms/1000)
    total_m = int(total_s/60)
    total_h = int(total_m/60)

    ms = round(total_ms % 1000)
    s = round(total_s % 60)
    m = round(total_m % 60)
    h = round(total_h % 24)

    timestamp = "{0:02d}:{1:02d}:{2:02d}:{3:03d}".format(h, m, s, ms)
    return timestamp


def frameindex_to_timestamp(frame_amount, fps):
    """Helper function to convert an amount of frames into a formatted timestamp."""
    # cv2's VideoCapture class provides a frame count property (cv2.CAP_PROP_FRAME_COUNT)
    # but not a duration property. However, as it is easier to think of video in terms of
    # duration/timestamps, it's helpful to convert back and forth.
    milliseconds = (frame_amount / fps)*1000
    timestamp = ms_to_timestamp(milliseconds)

    return timestamp


def timestamp_to_frameindex(timestamp, fps):
    """Helper function to convert formatted timestamp into an amount of frames."""
    # cv2's VideoCapture class provides a frame count property (cv2.CAP_PROP_FRAME_COUNT)
    # but not a duration property. However, as it is easier to think of video in terms of
    # duration/timestamps, it's helpful to convert back and forth.
    time = timestamp.split(":")
    seconds = (float(time[0])*60*60 +
               float(time[1])*60 +
               float(time[2]) +
               float(time[3])/1000)
    frame_amount = int(round(seconds * fps))
    return frame_amount


class FrameStack:
    """Class for storing, describing, and manipulating a stack of frames from a video file.

    Attributes: src_filename, src_directory, src_fps, src_framecount, src_height, src_width
                stack, indices, timestamps
    Methods: read_frames, save_frames, save_frames_info, load_frames, load_frames_info
    """
    def __init__(self, video_directory, filename, stack_size, desired_fps=False):
        # Check validity of filepath
        video_filepath = video_directory + filename
        if not os.path.isfile(video_filepath):
            raise Exception("[!] Filepath does not point to valid video file.")

        # Get attributes from source file that is associated with FrameStack.
        self.src_filename = filename
        self.src_directory = video_directory

        self.stream = cv2.VideoCapture("{}/{}".format(self.src_directory, self.src_filename))
        if not self.stream.isOpened():
            raise Exception("[!] Video file could not be opened to read frames. Check file path.")
        else:
            self.src_fps = self.stream.get(cv2.CAP_PROP_FPS)
            self.src_framecount = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
            self.src_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.src_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.src_codec = self.stream.get(cv2.CAP_PROP_FOURCC)

        # Initialize attributes of FrameStack
        self.stack = [None]*stack_size
        self.indices = [None]*stack_size
        self.timestamps = [' ']*stack_size
        self.frames_read = 0
        if not desired_fps:
            self.fps = self.src_fps
        else:
            self.fps = desired_fps

    def read_frame(self, delay=0):
        """Stores a new frame into index 0 of frame stack."""
        if not self.stream.isOpened():
            raise Exception("[!] Video stream is not open. Cannot read new frames.")

        # Shift frames in frame stack
        np.roll(self.stack, 1, axis=0)
        np.roll(self.indices, 1, axis=0)
        np.roll(self.timestamps, 1, axis=0)

        # Fetch new frame and update its attributes
        self.indices[0] = int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))
        self.timestamps[0] = (ms_to_timestamp(self.stream.get(cv2.CAP_PROP_POS_MSEC)))
        success, frame = self.stream.read()
        self.stack[0] = np.array(frame)
        if success:
            self.frames_read += 1

        # Increments position for next read frame (for skipping frames)
        for i in range(delay):
            self.stream.grab()

        return success

    def save_frame(self, base_save_directory, minute_flag=True, index=0):
        # TODO: Write a proper docstring for the save_frames() method.
        """Saves a set of frames to a """
        if minute_flag:
            time = self.timestamps[index].split(":")
            save_directory = base_save_directory+"/"+time[0]+":"+time[1]
        else:
            save_directory = base_save_directory+"/bulk-extraction"

        if not os.path.isdir(save_directory):
            try:
                os.makedirs(save_directory)
                sleep(0.5)  # Sometimes frame 0 won't be saved if delay is omitted
            except OSError:
                print("[!] Creation of the directory %s failed." % save_directory)

        try:
            cv2.imwrite("{0}/frame{1}_{2}.jpg".format(save_directory,
                                                      self.indices[index],
                                                      self.timestamps[index]),
                        self.stack[index])
        except Exception as e:
            print("[!] Image saving failed due to: {0}".format(str(e)))

    def convert_grayscale(self, index=0):
        # TODO: Write a proper docstring for the convert_grayscale() method.
        try:
            self.stack[index] = cv2.cvtColor(self.stack[index], cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print("[!] Frame conversion failed due to: {0}".format(str(e)))

    def crop_frame(self, corners, index=0):
        # TODO: Write a proper docstring for the crop_frames_rect() method.
        try:
            self.stack[index] = self.stack[index][corners[0][1]:corners[1][1], corners[0][0]:corners[1][0]]
        except Exception as e:
            print("[!] Frame cropping failed due to: {0}".format(str(e)))

    def resize_frame(self, scale_percent, index=0):
        s = scale_percent/100
        self.stack[index] = cv2.resize(self.stack[index],
                                       (round(self.stack[index].shape[1]*s), round(self.stack[index].shape[0]*s)),
                                       interpolation=cv2.INTER_AREA)
