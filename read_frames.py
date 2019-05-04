import cv2 as cv
import numpy as np

class frameObject:
    """Class object for storing frame image data and corresponding frame descriptors.

    Attributes:
        -frame:     ndarray of frames read from input video file.
        -index:     List of indices (from original video file) corresponding to each frame.
        -timestamp: List of timestamps (from original video file) corresponding to each frame.
    """
    def __init__(self, num_frames, height, width):
        self.frame = np.empty(num_frames, height, width, 3)
        self.index = np.empty(num_frames)
        self.timestamp = np.empty(num_frames)


def read_frames(input_video, start_frame=0, end_frame='default', sequence_type='reg', sample_rate='default'):
    """Returns a set of frames from a provided input video.

    Input arguments:
        -input_video:   Path to video file.
        -start_frame:   Index of frame at the start of the range of frames to read. ('0' = start of file)
        -end_frame:     Index of frame at the end of the range of frames to read. ('default' = end of file)
        -sequence_type: The pattern for how read frames are spaced. ('reg' = regularly spaced)
        -sample_rate:   The rate at which frames are saved from the video file. ('default' = input video frame-rate.)

    Output arguments:
        -frames:     ndarray of frames read from input video file.
        -indices:    List of indices (from original video file) corresponding to each frame.
        -timestamps: List of timestamps (from original video file) corresponding to each frame.
    """

