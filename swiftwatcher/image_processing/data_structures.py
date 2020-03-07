"""
    Contains data structures used to store video/image data. Used to
    cache groups of frames (for algorithms such as RPCA), and to store
    multiple versions of one frame at a time (to examine the state of
    a frame at various intermediate processing stages).
"""

from collections import OrderedDict, deque

from pathlib import Path
import swiftwatcher.image_processing.image_filtering as img
import cv2
import math


class Segment:
    """Class for representing a segment found within a frame. Stores
    various attributes of the segment, as well as its visual
    representation. This information is used to analyze the segments."""

    def __init__(self, regionprops, frame_number, timestamp, segment_image):
        self.parent_frame_number = frame_number
        self.parent_timestamp = timestamp
        self.segment_image = segment_image
        self.segment_history = []
        self.status = None

        for a in dir(regionprops):
            if not a.startswith('_'):
                setattr(self, a, getattr(regionprops, a, None))


class Frame:
    """Class for storing a frame from a video, as well as processed
    versions of that frame and its various properties."""

    src_video = None

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

    def set_segments(self, regionprops_list, segment_images):
        self.segments = [Segment(rp, self.frame_number, self.timestamp, seg)
                         for rp, seg in zip(regionprops_list, segment_images)]

    def export_segments(self, min_seg_size, crop_region, export_dir):
        if not export_dir.exists():
            Path.mkdir(export_dir, parents=True)

        color_img = self.processed_frames["crop"]
        for segment in self.segments:
            name_str = '"{}"_{}_{}_{}.png'.format(self.src_video,
                                                  self.frame_number,
                                                  segment.label,
                                                  len(self.segments))
            bbox = list(segment.bbox)

            # Bounding box provided by RegionProps: [H1, W1,   H2, W2]
            # My convention:                       [(W1, H1), (W2, H2)]
            # TODO: Fix all usages of my old convention in swiftwatcher
            crop = [crop_region[0][1], crop_region[0][0],
                    crop_region[1][1], crop_region[1][0]]

            # Export image highlighting area of segment
            overlay = color_img.copy()
            output = color_img.copy()
            alpha = 0.6
            cv2.rectangle(overlay, (bbox[1], bbox[0]), (bbox[3], bbox[2]),
                          (0, 0, 255), -1)
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
            if not Path.exists(export_dir / "overlay"):
                Path.mkdir(export_dir / "overlay")
            cv2.imwrite(str(export_dir / "overlay" / name_str), output)

            # Expand segment bbox to min seg size (keeping centered)
            dimensions = (bbox[2]-bbox[0], bbox[3]-bbox[1])
            if dimensions[0] < min_seg_size[0]:
                diff = min_seg_size[0] - dimensions[0]
                bbox[0] -= math.floor(diff/2)
                bbox[2] += math.ceil(diff/2)
            if dimensions[1] < min_seg_size[1]:
                diff2 = min_seg_size[1] - dimensions[1]
                bbox[1] -= math.floor(diff2 / 2)
                bbox[3] += math.ceil(diff2 / 2)

            # Extract segment image from full frame (not cropped image)
            bbox_f = [bbox[0]+crop[0], bbox[1]+crop[1],
                      bbox[2]+crop[0], bbox[3]+crop[1]]
            color_img_full = self.frame
            color_seg = color_img_full[bbox_f[0]:bbox_f[2],
                                       bbox_f[1]:bbox_f[3]]

            # Write segment to file
            cv2.imwrite(str(export_dir / name_str), color_seg)


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

    def store_segmented_queue(self, regionprops_lists, segment_image_list):
        for pos, (regionprops_list, segment_images) \
                in enumerate(zip(regionprops_lists, segment_image_list)):
            self[pos].set_segments(regionprops_list, segment_images)

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

        cropped_frames = [img.crop_frame(frame, crop_region)
                          for frame in self.get_queue()]
        self.store_processed_queue(cropped_frames, "crop")

        # resized_frames = [img.resize_frame(frame, resize_dim)
        #                        for frame in self.get_last_processed_queue()]
        # self.store_processed_queue(resized_frames, "resize")

        grayscale_frames = [img.convert_grayscale(frame)
                            for frame in self.get_last_processed_queue()]
        self.store_processed_queue(grayscale_frames, "grayscale")

    def segment_queue(self, min_seg_size, crop_region):
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
        segment_images = [img.extract_segment_images(regionprops_list,
                                                     frame, min_seg_size,
                                                     crop_region)
                          for frame, regionprops_list
                          in zip(self.get_queue(), regionprops_lists)]
        self.store_segmented_queue(regionprops_lists, segment_images)
