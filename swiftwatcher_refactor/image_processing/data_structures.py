"""
    Contains data structures used to store video/image data. Used to
    cache groups of frames (for algorithms such as RPCA), and to store
    multiple versions of one frame at a time (to examine the state of
    a frame at various intermediate processing stages).
"""

from collections import OrderedDict, deque

import swiftwatcher_refactor.image_processing.image_filtering as img


class Segment:
    """Class for representing a segment found within a frame. Stores
    various attributes of the segment, as well as its visual
    representation. This information is used to analyze the segments."""

    def __init__(self, regionprops, frame_number, timestamp):
        self.parent_frame_number = frame_number
        self.parent_timestamp = timestamp
        self.segment_image = None
        self.segment_history = []
        self.status = None

        for a in dir(regionprops):
            if not a.startswith('_'):
                setattr(self, a, getattr(regionprops, a, None))


class Frame:
    """Class for storing a frame from a video, as well as processed
    versions of that frame and its various properties."""

    def __init__(self, frame=None, frame_number=-1, timestamp="00:00:00.000"):
        self.frame_number = frame_number
        self.timestamp = timestamp

        self.frame = frame
        self.processed_frames = OrderedDict()
        self.segments = []

        if frame_number < 0:
            self.null = True
        else:
            self.null = False
        
    def get_frame(self):
        return self.frame
        
    def get_processed_frame(self, process_name):
        return self.processed_frames[process_name]

    def get_num_segments(self):
        return len(self.segments)

    def set_segments(self, regionprops_list):
        self.segments = [Segment(rp, self.frame_number, self.timestamp)
                         for rp in regionprops_list]


class FrameQueue(deque):
    """Class which extends Python's collections' deque class, adding
    methods specifically for handling Frame objects."""

    def __init__(self, queue_size=21):
        deque.__init__(self, maxlen=queue_size)

        self.frames_read = 0
        self.frames_processed = 0

    def is_empty(self):
        if len(self) == 0:
            return True
        else:
            return False

    def push_frame(self, input_frame, frame_number, timestamp):
        new_frame = Frame(input_frame, frame_number, timestamp)
        super(FrameQueue, self).appendleft(new_frame)
        self.frames_read += 1

    def push_list_of_frames(self, frame_list, frame_number_list,
                            timestamp_list):
        for frame, frame_number, timestamp \
                in zip(frame_list, frame_number_list, timestamp_list):
            self.push_frame(frame, frame_number, timestamp)

    def pop_frame(self):
        popped_frame = super(FrameQueue, self).pop()

        if popped_frame.null is False:
            self.frames_processed += 1

        return popped_frame

    def store_processed_queue(self, processed_frame_list, process_name):
        for pos, frame in enumerate(processed_frame_list):
            self[pos].processed_frames[process_name] = frame

    def store_segmented_queue(self, regionprops_lists):
        for pos, regionprops_list in enumerate(regionprops_lists):
            self[pos].set_segments(regionprops_list)

    def get_queue(self):
        return [frame_obj.frame for frame_obj in self]

    def get_processed_queue(self, process_name):
        return [frame_obj.processed_frames[process_name] for frame_obj in self]

    def get_last_processed_queue(self):
        # next(reversed()) accesses the last entry in an OrderedDict
        return [next(reversed(frame_obj.processed_frames.values()))
                for frame_obj in self]

    def preprocess_queue(self, crop_region, resize_dim):
        """Apply image filtering methods to preprocess every frame in
        queue, storing every stage individually."""

        grayscale_frames = [img.convert_grayscale(frame)
                            for frame in self.get_queue()]
        self.store_processed_queue(grayscale_frames, "grayscale")

        cropped_frames = [img.crop_frame(frame, crop_region)
                          for frame in self.get_last_processed_queue()]
        self.store_processed_queue(cropped_frames, "crop")

        preprocessed_frames = [img.resize_frame(frame, resize_dim)
                               for frame in self.get_last_processed_queue()]
        self.store_processed_queue(preprocessed_frames, "resize")

    def segment_queue(self):
        """Apply image filtering methods to segment every frame in
        queue, storing every stage individually."""

        rpca_frames = img.rpca(self.get_last_processed_queue())
        self.store_processed_queue(rpca_frames, "RPCA")

        bilateral_frames = [img.bilateral_blur(frame, 7, 15, 1)
                            for frame in self.get_last_processed_queue()]
        self.store_processed_queue(bilateral_frames, "bilateral")

        thresh_frames = [img.thresh_to_zero(frame, 15)
                         for frame in self.get_last_processed_queue()]
        self.store_processed_queue(thresh_frames, "thresh_15")

        opened_frames = [img.grayscale_opening(frame, (3, 3))
                         for frame in self.get_last_processed_queue()]
        self.store_processed_queue(opened_frames, "opened")

        labeled_frames = [img.cc_labeling(frame, 4)
                          for frame in self.get_last_processed_queue()]
        self.store_processed_queue(labeled_frames, "cc_labeling")

        regionprops_lists = [img.get_segment_properties(frame)
                             for frame in self.get_last_processed_queue()]
        self.store_segmented_queue(regionprops_lists)
