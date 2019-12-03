"""
    Contains functionality to track segments between frames, while also
    determining when a segment has appeared or disappeared within a
    frame.
"""

import swiftwatcher_refactor.image_processing.data_structures as ds


class SegmentTracker:
    def __init__(self, roi_mask):
        self.current_frame = None
        self.cached_frame = ds.Frame()  # Empty frame object

        # Used when detecting a "swift entered chimney" event
        self.roi_mask = roi_mask
        self.detected_events = None

    def get_current_frame(self):
        return self.current_frame

    def get_cached_frame(self):
        return self.cached_frame

    def set_current_frame(self, frame):
        self.current_frame = frame

    def cache_current_frame(self):
        self.cached_frame = self.current_frame
