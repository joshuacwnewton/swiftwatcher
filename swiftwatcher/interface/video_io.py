"""
    Contains functionality for video frame I/O.
"""

import pandas as pd
import numpy as np
import cv2
import h5py


class FrameReader:
    """Base class for reading frames from a video source."""

    def __init__(self):
        self.fps = 0
        self.start_frame = 0
        self.end_frame = 0
        self.total_frames = 0
        self.next_frame_number = 0

        self.frame_shape = (0, 0, 0)
        self.last_read_frame = None
        self.frames_read = 0
        self.read_errors = 0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "read_frame"):
            raise NotImplementedError("Derived FrameReader must implement "
                                      "read_frame() method.")

    def get_frame(self, frame_number=None):
        """Returns frame, frame_number, and timestamp while also
        handling read errors."""

        if frame_number is None:
            frame_number = self.next_frame_number

        if not self.start_frame <= frame_number <= self.end_frame:
            # Dummy values for if invalid frame is requested
            frame = np.zeros(self.frame_shape).astype(np.uint8)
            frame_number = -1
            timestamp = "00:00:00.000"

        else:
            # Subclass must implement this method
            frame = self.read_frame(frame_number)
            timestamp = self.frame_number_to_timestamp(frame_number)

            if frame is None:
                frame = self.last_read_frame
                self.read_errors += 1
            else:
                self.frame_shape = frame.shape
                self.last_read_frame = frame
                self.frames_read += 1

        return frame, frame_number, timestamp

    def get_n_frames(self, n):
        """Calls get_frame in batches of N, returning as lists."""

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


class HDF5Reader(FrameReader):
    """Subclass using HDF5 data container as frame source."""

    def __init__(self, filepath, start=0, end=0):
        super().__init__()

        # Set file object using filepath for reading frames
        self.filepath = filepath
        self.hdf5_file = h5py.File(str(filepath), "r")
        self.dset = self.hdf5_file["VideoFrames"]

        # Attempt to read attributes from HDF5 group or dataset
        if len(self.hdf5_file.attrs) > 0:
            attrs = self.hdf5_file.attrs
        elif len(self.dset.attrs) > 0:
            attrs = self.dset.attrs
        else:
            raise RuntimeError("Passed HDF5 dataset does not contain attrs.")

        self.fps = attrs.get("CAP_PROP_FPS")

        # Set start/end frame numbers, or default if not properly passed
        self.start_frame = start
        if end > 0:
            self.end_frame = end
        else:
            self.end_frame = int(attrs.get("CAP_PROP_FRAME_COUNT"))

        self.next_frame_number = self.start_frame
        self.total_frames = self.end_frame - self.start_frame

    def read_frame(self, frame_number, increment=True):
        """Read frame from HDF5 container, fulfills constraint from
        base class."""

        try:
            encoded_frame = self.dset[frame_number]
            frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
        except ValueError as e:
            print(e)
            print("HDF5Reader returning empty frame instead.")
            frame = None

        if increment:
            self.next_frame_number += 1

        return frame


class VideoReader(FrameReader):
    """Subclass using OpenCV's VideoCapture as frame source."""

    def __init__(self, filepath, end):
        super().__init__()

        # Set file object using filepath for reading frames
        self.filepath = filepath
        self.vid_cap = cv2.VideoCapture(str(filepath))
        self.vid_cap.grab()  # Load first frame so retrieve() won't fail

        self.fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
        self.start_frame = 0
        if end > 0:
            self.end_frame = end
        else:
            self.end_frame = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.next_frame_number = self.start_frame
        self.total_frames = self.end_frame - self.start_frame

    def read_frame(self, frame_number, increment=True):
        """Read frame from video file, fulfills constraint from base
        class."""

        _, frame = self.vid_cap.retrieve()

        if increment:
            self.vid_cap.grab()
            self.next_frame_number += 1

        return frame
