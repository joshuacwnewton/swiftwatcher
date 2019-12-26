"""
    Contains functionality for video I/O, as well as video frame I/O.
"""

import sys
from pathlib import Path
from glob import glob
import re

import json
from datetime import date

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
    vidcap = cv2.VideoCapture(str(filepath))
    retval, frame = vidcap.read()
    vidcap.release()

    return frame


def get_frame_from_file(path, frame_number):
    """Generic function for reading a numbered frame from a file.
    Assumes that frames are stored within a subfolder. Ideally, should
    be rewritten w/ regular expressions to make subfolders optional."""

    frame_list = glob(str(path/"*"/("*_" + str(frame_number) + "_*.png")))
    frame = cv2.imread(frame_list[0])

    return frame


class FrameReader:
    """An alternative to OpenCV's VideoCapture class. Fetches frames
    from image files rather than a video file. This allows tests
    to be run on specific sequences of frames if they're extracted
    ahead of time, which is useful for hours-long video files."""

    def __init__(self, frame_dir, fps, start, end):
        self.frame_dir = frame_dir
        self.frame_path_list = glob(str(self.frame_dir/"**"/"*.png"),
                                    recursive=True)
        p = re.compile(r'.*_(\d+)_.*')
        self.frame_path_dict = {m.group(1): m.group(0) for m in
                                [p.match(s) for s in self.frame_path_list]}
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
        frames, frame_numbers, timestamps = [], [], []
        for _ in range(n):
            frame, frame_number, timestamp = self.get_frame()

            frames.append(frame)
            frame_numbers.append(frame_number)
            timestamps.append(timestamp)

        return frames, frame_numbers, timestamps

    def frame_number_to_timestamp(self, frame_number):
        total_s = frame_number / self.fps
        timestamp = pd.Timestamp("00:00:00.000") + pd.Timedelta(total_s, 's')
        timestamp = timestamp.round(freq='us')

        return timestamp


class VideoReader(cv2.VideoCapture):
    def __init__(self, video_filepath):
        super(VideoReader, self).__init__(str(video_filepath))

        self.total_frames = int(self.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames_read = 0
        self.frame_shape = None

        self.last_frame_cache = None
        self.read_errors = 0

    def get_frame(self):
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


###############################################################################
#               RESEARCH EXPERIMENTATION FUNCTIONS BEGIN HERE                 #
###############################################################################


def get_corners_from_file(parent_directory):
    """Non-GUI alternative to get_video_attributes for time saving."""

    with open(str(parent_directory / "attributes.json")) as json_file:
        video_attributes = json.load(json_file)

        # Convert from string to individual integer values
        video_attributes["corners"] \
            = [(int(video_attributes["corners"][0][0]),
                int(video_attributes["corners"][0][1])),
               (int(video_attributes["corners"][1][0]),
                int(video_attributes["corners"][1][1]))]

    return video_attributes["corners"]


def generate_test_dir(parent_dir):
    """Generate test directory based on the following scheme:
        parent_dir/<today's date>/<ID of last test + 1>

    If no test has been run today, set ID to 1."""

    # Set base testing directory to today's date
    date_dir = parent_dir / str(date.today())

    if not date_dir.exists():
        # Date directory doesnt exist, so must be first test run today
        test_dir = date_dir / "1"

    else:
        # Fetch names of all subdirectories in date_dir, then get the max
        last_test_id = max([int(path.stem) for path in
            [Path(path_str) for path_str in glob(str(date_dir / "*/"))]])

        # Set test directory to last test incremented by one
        test_dir = date_dir / str(last_test_id + 1)

    return test_dir