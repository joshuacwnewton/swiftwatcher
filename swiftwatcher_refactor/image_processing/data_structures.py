"""
    Contains data structures used to store video/image data. Used to
    cache groups of frames (for algorithms such as RPCA), and to store
    multiple versions of one frame at a time (to examine the state of
    a frame at various intermediate processing stages).
"""

from collections import OrderedDict, deque

import swiftwatcher_refactor.image_processing.image_filtering as img
import swiftwatcher_refactor.io.video_io as vio


class Frame:
    """Class for storing a frame from a video, as well as processed
    versions of that frame and its various properties."""

    def __init__(self, frame, frame_number):
        self.frame_number = frame_number
        self.timestamp = None

        self.frame = frame
        self.processed_frames = OrderedDict()
        self.segment_properties = None


class FrameQueue(deque):
    """Class which extends Python's stdlib queue class, adding methods
    specifically for handling Frame objects."""

    def __init__(self, src_path, total_frames=0, start_frame=0, queue_size=21):
        deque.__init__(self, maxlen=queue_size)

        self.src_path = src_path
        if self.src_path.is_file():
            self.stream = None
            self.start_frame = start_frame
            self.total_frames = total_frames

        else:
            self.start_frame = start_frame
            self.total_frames = total_frames

        self.frames_read = 0
        self.frames_processed = 0

    def set_frame(self, input_frame, frame_number):
        new_frame = Frame(input_frame, frame_number)
        super(FrameQueue, self).append(new_frame)
        self.frames_read += 1

    def get_frame(self, pos=-1):
        return self[pos].frame

    def set_processed_frame(self, input_frame, process_name, pos=-1):
        self[pos].processed_frames[process_name] = input_frame

    def get_processed_frame(self, process_name, pos=-1):
        return self[pos].processed_frames[process_name]

    def fill_queue(self):
        for i in range(self.maxlen):
            # Fetch frame from file if directory was passed
            if self.src_path.is_dir():
                frame_number = self.start_frame + self.frames_read
                frame = vio.get_frame_from_file(self.src_path, frame_number)

            # Fetch frame from stream if video file was passed
            elif self.src_path.is_file():
                do_file_stream_reading = None

            self.set_frame(frame, frame_number)

    def preprocess_queue(self, crop_region, resize_dim):
        for i in range(len(self)):
            preproc_frame = img.preprocess_frame(self.get_frame(i),
                                                 crop_region, resize_dim)
            self.set_processed_frame(preproc_frame, "preprocessed", pos=i)

    def segment_queue(self):
        test = None
