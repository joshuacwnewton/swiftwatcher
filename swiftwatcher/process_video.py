import os
import glob
import collections
from time import sleep
import cv2
import numpy as np
from utils.rpca_ialm import inexact_augmented_lagrange_multiplier


class FrameQueue:
    """Class for storing, describing, and manipulating frames from video file using a FIFO queue.
    Essentially a deque with additional attributes and methods specific to video analysis.
    Queue is size 1 by default. Increase queue size for analysis requiring multiple sequential frames.

    Attributes (frame queue):  queue, framenumbers, timestamps, frames_read
    Attributes (source video): src_filename, src_directory, src_fps, src_framecount,
                               src_height, src_width, src_codec
    Attributes (frames):       height, width, delay, fps

    Methods (File I/O): load_frame_from_video, load_frame_from_file, save_frame_to_file,
    Methods (Frame processing): convert_grayscale, segment_frame, crop_frame, resize_frame,
                                frame_to_column, rpca_decomposition
    """
    def __init__(self, video_directory, filename, queue_size=1, desired_fps=False):
        # Initialize queue attributes
        self.queue = collections.deque([], queue_size)
        self.framenumbers = collections.deque([], queue_size)
        self.timestamps = collections.deque([], queue_size)
        self.frames_read = 0

        # Check validity of filepath
        video_filepath = video_directory + filename
        if not os.path.isfile(video_filepath):
            raise Exception("[!] Filepath does not point to valid video file.")

        # Open source video file and store its immutable attributes
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

        # Initialize mutable frame/video attributes
        self.delay = round(self.src_fps/self.fps) - 1  # Needed to subsample video
        self.height = self.src_height  # Separate because cropping may change dimensions
        self.width = self.src_width    # Separate because cropping may change dimensions
        if not desired_fps:
            self.fps = self.src_fps
        else:
            self.fps = desired_fps

    def load_frame_from_video(self):
        """Insert next frame from video stream into left side (index 0) of frame queue."""
        if not self.stream.isOpened():
            raise Exception("[!] Video stream is not open. Cannot read new frames.")

        # Fetch new frame and update its attributes
        self.framenumbers.appendleft(int(self.stream.get(cv2.CAP_PROP_POS_FRAMES)))
        self.timestamps.appendleft((ms_to_timestamp(self.stream.get(cv2.CAP_PROP_POS_MSEC))))
        success, frame = self.stream.read()
        if success:
            self.queue.appendleft(np.array(frame))
            self.frames_read += 1

        # Increments position for next read frame (for skipping frames)
        for i in range(self.delay):
            self.stream.grab()

        return success

    def load_frame_from_file(self, base_save_directory, frame_number, folder_name=None):
        """Insert specified frame from file into left side (index 0) of frame queue."""
        # Update frame attributes
        self.framenumbers.appendleft(frame_number)
        timestamp = framenumber_to_timestamp(frame_number, self.fps)
        self.timestamps.appendleft(timestamp)

        # Set appropriate directory
        if not folder_name:
            time = self.timestamps[0].split(":")
            save_directory = base_save_directory+"/"+time[0]+":"+time[1]
        else:
            save_directory = base_save_directory+"/"+folder_name

        # Load frame from file
        file_paths = glob.glob("{}/frame{}_{}*".format(save_directory, frame_number, timestamp))
        frame = cv2.imread(file_paths[0])
        self.queue.appendleft(np.array(frame))

        if not frame.size == 0:
            success = True
            self.frames_read += 1
        else:
            success = False

        return success

    def save_frame_to_file(self, base_save_directory, frame=None, index=0,
                           folder_name=None, file_prefix="", file_suffix="",
                           scale=100):
        """Save an individual frame to an image file. If frame itself is not provided,
           Frame will be pulled from frame_queue at specified index."""

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

        # Extract frame from specified queue object as fallback
        if frame is None:
            frame = self.queue[index]

        # Resize frame for viewing convenience
        s = scale/100
        if s is not 1:
            resized_frame = cv2.resize(frame,
                                       (round(frame.shape[1]*s), round(frame.shape[0]*s)),
                                       interpolation=cv2.INTER_AREA)

        # Write frame to image file within save_directory
        try:
            cv2.imwrite("{0}/{1}frame{2}_{3}{4}.jpg".format(save_directory,
                                                            file_prefix,
                                                            self.framenumbers[index],
                                                            self.timestamps[index],
                                                            file_suffix),
                        resized_frame)
        except Exception as e:
            print("[!] Image saving failed due to: {0}".format(str(e)))

    def convert_grayscale(self, index=0):
        """Convert to grayscale a frame at specified index of FrameQueue"""
        try:
            self.queue[index] = cv2.cvtColor(self.queue[index], cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print("[!] Frame conversion failed due to: {0}".format(str(e)))

    def crop_frame(self, corners, index=0):
        """Crop frame at specified index of FrameQueue."""
        try:
            self.queue[index] = self.queue[index][corners[0][1]:corners[1][1], corners[0][0]:corners[1][0]]
        except Exception as e:
            print("[!] Frame cropping failed due to: {0}".format(str(e)))

        # Update frame attributes if necessary
        height = corners[1][1] - corners[0][1]
        width = corners[1][0] - corners[0][0]
        if height is not self.height:
            self.height = height
        if width is not self.width:
            self.width = width

    def resize_frame(self, scale_percent, index=0):
        s = scale_percent/100
        self.queue[index] = cv2.resize(self.queue[index],
                                       (round(self.queue[index].shape[1]*s), round(self.queue[index].shape[0]*s)),
                                       interpolation=cv2.INTER_AREA)

    def frame_to_column(self, index=0):
        """Reshapes an NxM frame into an (N*M)x1 column vector."""
        self.queue[index] = np.squeeze(np.reshape(self.queue[index],
                                                  (self.width*self.height, 1)))

    def rpca_decomposition(self, index):
        """Decompose set of images into corresponding low-rank and sparse images.
        Method expects images to have been reshaped to matrix of column vectors.

        NOTE: Currently this code inputs a batch of frames (the entire queue) and outputs
        the decomposition only for a specific frame. This ensures the best accuracy (uses
        context from before and after specified frame) but is inefficient.
        Could be tweaked in the future."""

        # np.array alone would give an e.g. 20x12500x1 matrix. Adding
        # np.transpose and np.squeeze yields 12500x20 (i.e. matrix of column vectors)
        matrix = np.squeeze(np.transpose(np.array(self.queue)))

        # Algorithm for the IALM approximation of Robust PCA method.
        lr_columns, s_columns = inexact_augmented_lagrange_multiplier(matrix, verbose=False)

        # Slice key frame from low rank, sparse results and reshape back into image
        lr_image = np.reshape(lr_columns[:, index], (self.height, self.width))
        s_image = np.reshape(s_columns[:, index], (self.height, self.width))

        return lr_image, s_image


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


def extract_frames(video_directory, filename, queue_size=1, save_directory=None):
    """Helper function to extract individual frames one at a time 
       from video file and save them to image files for future reuse."""

    # Default save directory chosen to be identical to filename
    if not save_directory:
        save_directory = video_directory + os.path.splitext(filename)[0]

    print("[========================================================]")
    print("[*] Reading frames... (This may take a while!)")

    frame_queue = FrameQueue(video_directory, filename, queue_size)
    while frame_queue.frames_read < frame_queue.src_framecount:
        # Attempt to read frame from video into queue object
        success = frame_queue.load_frame_from_video()

        # Save frame to file if read correctly
        if success:
            frame_queue.save_frame_to_file(save_directory)
        else:
            raise Exception("read_frame() failed before expected end of file.")

        # Status updates
        if frame_queue.frames_read % 1000 == 0:
            print("[-] {}/{} frames successfully processed.".format(frame_queue.frames_read,
                                                                    frame_queue.src_framecount))
    frame_queue.stream.release()

    print("[========================================================]")
    print("[-] Extraction complete. {} total frames extracted.".format(frame_queue.frames_read))