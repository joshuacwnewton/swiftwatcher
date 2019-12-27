"""
    Contains functionality for video I/O, as well as video frame I/O.
"""

import sys
from pathlib import Path
from glob import glob
import re

import pandas as pd
import numpy as np
import cv2


###############################################################################
#                        VALIDATION FUNCTIONS BEGIN HERE                      #
###############################################################################


def validate_directories(dirpaths):
    """Ensure that directory path points to valid directory """
    if type(dirpaths) is not list:
        dirpaths = [dirpaths]

    for dirpath in dirpaths:
        dirpath = Path(dirpath)
        if not dirpath.is_dir():
            sys.stderr.write("Error: {} doesn't point to directory."
                             .format(dirpath))
            sys.exit()


def validate_filepaths(filepaths):
    """Ensure that file path points to a valid file."""

    if type(filepaths) is not list:
        filepaths = [filepaths]

    for filepath in filepaths:
        filepath = Path(filepath)

        if not Path.is_file(filepath):
            sys.stderr.write("[!] Error: {} does not point to a valid file."
                             .format(filepath.name))
            sys.exit()


def validate_video_files(video_filepaths):
    """Ensure that frames can be read from video file."""

    if type(video_filepaths) is not list:
        video_filepaths = [video_filepaths]

    for video_filepath in video_filepaths:
        vidcap = cv2.VideoCapture(str(video_filepath))
        retval, _ = vidcap.read()

        if retval is False:
            sys.stderr.write("[!] Error: Unable to read frames from {}."
                             .format(video_filepath.name))
            sys.exit()

        vidcap.release()


def validate_video_filepaths(video_filepaths):
    """Basic checks on a given video filepath"""

    validate_filepaths(video_filepaths)
    validate_video_files(video_filepaths)


def validate_frame_order(start, end):
    """Ensure that ordering of start/end values are correct"""

    if not end > start > -1:
        sys.stderr.write("Error: Start/end values not correct."
                         " (non-zero with end > start).")
        sys.exit()


def validate_frame_file(frame_dir, frame_number):
    """Checks if a png file with 'frame_number' in the filename exists
    in a subfolder of 'frame_dir'."""

    if not glob(str(frame_dir/"*"/("*_"+str(frame_number)+"_*"+".png"))):
        sys.stderr.write("Error: Frame {} does not point to valid file."
                         .format(frame_number))
        sys.exit()


def validate_frame_range(frame_dir, start, end):
    """Validate if start and end frame numbers point to valid frame
    files."""

    validate_directories(frame_dir)
    validate_frame_order(start, end)
    validate_frame_file(frame_dir, start)
    validate_frame_file(frame_dir, end)


###############################################################################
#                  VIDEO/FRAME READING FUNCTIONS BEGIN HERE                   #
###############################################################################


def get_video_properties(filepath):
    """Store video properties in dictionary so video file/VidCap object
    aren't needed to access properties."""

    vidcap = cv2.VideoCapture(str(filepath))

    properties = {
        "fps": int(vidcap.get(cv2.CAP_PROP_FPS)),

        # Include full-video frame start/end dates
        "start": 0,
        "end": int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    }

    vidcap.release()

    return properties


def get_first_video_frame(filepath):
    """Fetch the first video frame from a video file (for use with e.g.
    determining an ROI for a static video feed.)"""

    vidcap = cv2.VideoCapture(str(filepath))
    retval, frame = vidcap.read()
    vidcap.release()

    return frame


class FrameReader:
    """Fetches frames from image files, rather than directly from a
    video file like OpenCV's VideoCapture class would. This avoids the
    limitation of having to process entire videos from the beginning.
    However, this requires the frames to have been extracted beforehand.
    Uses custom Frame objects defined in data_structures.py."""

    def __init__(self, frame_dir, fps, start, end):
        # Prefetch a mapping from frame numbers to frame filepaths
        self.frame_dir = frame_dir
        self.frame_path_list = glob(str(self.frame_dir/"**"/"*.png"),
                                    recursive=True)
        p = re.compile(r'.*_(\d+)_.*')
        self.frame_path_dict = {m.group(1): m.group(0) for m in
                                [p.match(s) for s in self.frame_path_list]}

        # Store video properties
        self.fps = fps
        self.start_frame = start
        self.end_frame = end
        self.total_frames = end - start + 1

        self.frames_read = 0
        self.next_frame_number = self.start_frame
        self.frame_shape = None

    def get_filepath(self, frame_number):
        return self.frame_path_dict[str(frame_number)]

    def get_frame(self):
        """Fetch frame from file if there are frames left, or a dummy
        file if there are no frames left."""

        if self.next_frame_number <= self.end_frame:
            frame_number = self.next_frame_number
            timestamp = self.frame_number_to_timestamp(frame_number)

            filepath = self.get_filepath(frame_number)
            frame = cv2.imread(filepath)

            if frame.data:
                self.frame_shape = frame.shape
                self.frames_read += 1
                self.next_frame_number += 1

        # This is for the case when frames are requested in batches of N, but
        # total_frames is not a multiple of N. In that case, self.end_frame
        # will eventually be exceeded, so return empty values.
        else:
            frame = np.zeros(self.frame_shape).astype(np.uint8)
            frame_number = -1
            timestamp = "00:00:00.000"

        return frame, frame_number, timestamp

    def get_n_frames(self, n):
        """Fetch frames from files in batches of N, with return values
        put into individual lists."""

        frames, frame_numbers, timestamps = [], [], []
        for _ in range(n):
            frame, frame_number, timestamp = self.get_frame()

            frames.append(frame)
            frame_numbers.append(frame_number)
            timestamps.append(timestamp)

        return frames, frame_numbers, timestamps

    def frame_number_to_timestamp(self, frame_number):
        """Simple conversion to get timestamp from frame number.
        Dependent on constant FPS assumption for source video file."""

        total_s = frame_number / self.fps
        timestamp = pd.Timestamp("00:00:00.000") + pd.Timedelta(total_s, 's')
        timestamp = timestamp.round(freq='us')

        return timestamp


class VideoReader(cv2.VideoCapture):
    """Extends OpenCV's built-in VideoCapture class with additional
    methods for batch processing, error handling, etc. Uses custom Frame
    objects defined in data_structures.py."""

    def __init__(self, video_filepath):
        super(VideoReader, self).__init__(str(video_filepath))

        self.total_frames = int(self.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames_read = 0
        self.frame_shape = None

        self.last_frame_cache = None
        self.read_errors = 0

    def get_frame(self):
        """Fetch frame from video if there are frames left, or a dummy
        file if there are no frames left."""
        if self.get(cv2.CAP_PROP_POS_FRAMES) <= self.total_frames:
            frame_number = int(self.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = self.ms_to_ts(self.get(cv2.CAP_PROP_POS_MSEC))
            success, frame = self.read()

            if success:
                self.frame_shape = frame.shape
                self.frames_read += 1
                self.last_frame_cache = frame
            else:
                # Use last frame, to prevent linked segments paths from
                # breaking if read errors occur in the middle of a video file
                frame = self.last_frame_cache
                self.read_errors += 1

        # This is for the case when frames are requested in batches of N, but
        # total_frames is not a multiple of N. In that case, self.end_frame
        # will eventually be exceeded, so return empty values.
        else:
            frame = np.zeros(self.frame_shape).astype(np.uint8)
            frame_number = -1
            timestamp = "00:00:00.000"

        return frame, frame_number, timestamp

    def get_n_frames(self, n):
        """Fetch frames from video in batches of N, with return values
        put into individual lists."""

        frames, frame_numbers, timestamps = [], [], []
        for _ in range(n):
            frame, frame_number, timestamp = self.get_frame()

            frames.append(frame)
            frame_numbers.append(frame_number)
            timestamps.append(timestamp)

        return frames, frame_numbers, timestamps

    def ms_to_ts(self, ms):
        timestamp = pd.Timestamp("00:00:00.000") + pd.Timedelta(ms, 'ms')

        timestamp = timestamp.round(freq='us')

        return timestamp


