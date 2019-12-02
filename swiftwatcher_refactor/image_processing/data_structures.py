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
    specifically for handling Frame objects. (Getters, setters, and
    image processing methods)."""

    def __init__(self, queue_size=21):
        deque.__init__(self, maxlen=queue_size)

        self.frames_read = 0
        self.frames_processed = 0

    def set_frame(self, input_frame, frame_number):
        new_frame = Frame(input_frame, frame_number)
        super(FrameQueue, self).append(new_frame)
        self.frames_read += 1

    def set_queue(self, frame_list, frame_number_list):
        for frame, frame_number in zip(frame_list, frame_number_list):
            self.set_frame(frame, frame_number)

    def set_processed_frame(self, input_frame, process_name, pos=-1):
        self[pos].processed_frames[process_name] = input_frame

    def set_processed_queue(self, frame_list, process_name):
        for pos, frame in enumerate(frame_list):
            self[pos].processed_frames[process_name] = frame

    def get_frame(self, pos=-1):
        return self[pos].frame

    def get_queue(self):
        return [frame_obj.frame for frame_obj in self]

    def get_processed_frame(self, process_name, pos=-1):
        return self[pos].processed_frames[process_name]

    def preprocess_queue(self, crop_region, resize_dim):
        grayscale_frames = [img.convert_grayscale(frame)
                            for frame in self.get_queue()]
        self.set_processed_queue(grayscale_frames, "grayscale")

        cropped_frames = [img.crop_frame(frame, crop_region)
                          for frame in grayscale_frames]
        self.set_processed_queue(cropped_frames, "crop")

        preprocessed_frames = [img.resize_frame(frame, resize_dim)
                               for frame in cropped_frames]
        self.set_processed_queue(preprocessed_frames, "resize")

    def segment_queue(self):
        test = None
