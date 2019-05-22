import os
import glob
import collections
import cv2
import numpy as np
from utils.rpca_ialm import inexact_augmented_lagrange_multiplier

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


def framenumber_to_timestamp(frame_number, fps):
    """Helper function to convert an amount of frames into a formatted timestamp."""
    # cv2's VideoCapture class provides a frame count property (cv2.CAP_PROP_FRAME_COUNT)
    # but not a duration property. However, as it is easier to think of video in terms of
    # duration/timestamps, it's helpful to convert back and forth.
    milliseconds = (frame_number / fps)*1000
    timestamp = ms_to_timestamp(milliseconds)

    return timestamp


def timestamp_to_framenumber(timestamp, fps):
    """Helper function to convert formatted timestamp into an amount of frames."""
    # cv2's VideoCapture class provides a frame count property (cv2.CAP_PROP_FRAME_COUNT)
    # but not a duration property. However, as it is easier to think of video in terms of
    # duration/timestamps, it's helpful to convert back and forth.
    time = timestamp.split(":")
    seconds = (float(time[0])*60*60 +
               float(time[1])*60 +
               float(time[2]) +
               float(time[3])/1000)
    frame_number = int(round(seconds * fps))
    return frame_number


class FrameStack:
    """Class for storing, describing, and manipulating frames from video file using a FIFO stack.
    Essentially a deque with additional attributes and methods specific to video analysis.
    Stack is size 1 by default. Increase stack size for analysis requiring multiple sequential frames.

    Attributes (source video): src_filename, src_directory, src_fps, src_framecount,
                              src_height, src_width, src_codec
    Attributes (frame stack): stack, framenumbers, timestamps, fps, delay

    Methods (File I/O): read_frame_from_video, save_frame_to_file, load_frame_from_file
    Methods (Frame processing): convert_grayscale, segment_frame, crop_frame, resize_frame
    """
    def __init__(self, video_directory, filename, stack_size=1, desired_fps=False):
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
        self.stack = collections.deque([], stack_size)
        self.framenumbers = collections.deque([], stack_size)
        self.timestamps = collections.deque([], stack_size)
        self.frames_read = 0
        if not desired_fps:
            self.fps = self.src_fps
        else:
            self.fps = desired_fps
        # Delay between frames, needed to subsample video at lower framerate
        self.delay = round(self.src_fps / self.fps) - 1

    def load_frame_from_video(self):
        """Insert next frame from video stream into left side (index 0) of frame stack."""
        if not self.stream.isOpened():
            raise Exception("[!] Video stream is not open. Cannot read new frames.")

        # Fetch new frame and update its attributes
        self.framenumbers.appendleft(int(self.stream.get(cv2.CAP_PROP_POS_FRAMES)))
        self.timestamps.appendleft((ms_to_timestamp(self.stream.get(cv2.CAP_PROP_POS_MSEC))))
        success, frame = self.stream.read()
        if success:
            self.stack.appendleft(np.array(frame))
            self.frames_read += 1

        # Increments position for next read frame (for skipping frames)
        for i in range(self.delay):
            self.stream.grab()

        return success

    def load_frame_from_file(self, base_save_directory, frame_number, folder_name=None):
        """Insert specified frame from file into left side (index 0) of frame stack."""
        # Update frame attributes
        self.framenumbers.appendleft(frame_number)
        self.timestamps.appendleft(framenumber_to_timestamp(frame_number, self.fps))

        # Set appropriate directory
        if not folder_name:
            time = self.timestamps[0].split(":")
            save_directory = base_save_directory+"/"+time[0]+":"+time[1]
        else:
            save_directory = base_save_directory+"/"+folder_name

        # Load frame from file
        file_paths = glob.glob("{}/frame{}*".format(save_directory, frame_number))
        frame = cv2.imread(file_paths[0])
        self.stack.appendleft(np.array(frame))

        if not frame.size == 0:
            success = True
            self.frames_read += 1
        else:
            success = False

        return success

    def save_frame_to_file(self, base_save_directory, index=0, folder_name=None):
        """Save an individual frame to an image file."""

        # By default, frames will be saved in a subfolder corresponding to HH:MM.
        # If specified, however, a custom subfolder can be chosen instead.
        if not folder_name:
            time = self.timestamps[index].split(":")
            save_directory = base_save_directory+"/"+time[0]+":"+time[1]
        else:
            save_directory = base_save_directory+"/"+folder_name

        # Create save directory if it does not already exist
        if not os.path.isdir(save_directory):
            try:
                os.makedirs(save_directory)
                sleep(0.5)  # Sometimes frame 0 won't be saved if delay is omitted
            except OSError:
                print("[!] Creation of the directory %s failed." % save_directory)

        # Write frame to image file within save_directory
        try:
            cv2.imwrite("{0}/frame{1}_{2}.jpg".format(save_directory,
                                                      self.framenumbers[index],
                                                      self.timestamps[index]),
                        self.stack[index])
        except Exception as e:
            print("[!] Image saving failed due to: {0}".format(str(e)))

    def convert_grayscale(self, index=0):
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

    def frame_to_column(self, index=0):
        """Reshape image matrix to single column vector."""
        self.stack[index] = np.reshape(self.stack[index], (-1, 1))

    def concatenate_stack(self):
        """Returns matrix comprised of all stack entries concatenated together"""
        return np.concatenate(self.stack, axis=1)

    def rpca_decomposition(self, matrix):
        low_rank, sparse = inexact_augmented_lagrange_multiplier(matrix)
        return low_rank, sparse

    def resize_frame(self, scale_percent, index=0):
        s = scale_percent/100
        self.stack[index] = cv2.resize(self.stack[index],
                                       (round(self.stack[index].shape[1]*s), round(self.stack[index].shape[0]*s)),
                                       interpolation=cv2.INTER_AREA)


def extract_frames(video_directory, filename, stack_size=1, save_directory=None):
    """Helper function to extract individual frames one at a time 
       from video file and save them to image files."""

    # Default save directory chosen to be identical to filename
    if not save_directory:
        save_directory = video_directory + os.path.splitext(filename)[0]

    print("[========================================================]")
    print("[*] Reading frames... (This may take a while!)")

    frame_stack = FrameStack(video_directory, filename, stack_size)
    while frame_stack.frames_read < frame_stack.src_framecount:
        # Attempt to read frame from video into stack object
        success = frame_stack.load_frame_from_video()

        # Save frame to file if read correctly
        if success:
            frame_stack.save_frame_to_file(save_directory)
        else:
            raise Exception("read_frame() failed before expected end of file.")

        # Status updates
        if frame_stack.frames_read % 1000 == 0:
            print("[-] {}/{} frames successfully processed.".format(frame_stack.frames_read,
                                                                    frame_stack.src_framecount))
    frame_stack.stream.release()

    print("[========================================================]")
    print("[-] Extraction complete. {} total frames extracted.".format(frame_stack.frames_read))