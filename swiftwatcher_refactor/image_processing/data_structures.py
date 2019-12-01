"""
    Contains data structures used to store video/image data. Used to
    cache groups of frames (for algorithms such as RPCA), and to store
    multiple versions of one frame at a time (to examine the state of
    a frame at various intermediate processing stages).
"""

from collections import OrderedDict
from queue import Queue


class Frame:
    """Class for storing a video's frame, as well as processed versions
    of that frame, and its various properties."""

    def __init__(self, frame):
        self.frame_number = None
        self.timestamp = None

        self.frame = frame
        self.processed_frames = OrderedDict()
        self.segment_properties = None

    def get_unprocessed_frame(self):
        return self.frame

    def set_processed_frame(self, frame, process_name):
        self.processed_frames[process_name] = frame


class FrameQueue(Queue):
    """Class which extends Python's stdlib queue class, with methods
    specifically for handling Frame objects."""

    def __init__(self, queue_size):
        Queue.__init__(self, maxsize=queue_size)

    def put(self, input_frame, block=True, timeout=None):
        super(FrameQueue, self).put(Frame(input_frame))

    def set_processed_frame(self, input_frame, process_name, pos):
        self.queue[pos].set_processed_frame(input_frame, process_name)
