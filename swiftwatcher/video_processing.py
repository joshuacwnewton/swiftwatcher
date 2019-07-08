# Stdlib imports
import os
import glob
import collections
import math
from time import sleep

# Imports used in numerous stages
import cv2
import numpy as np
import utils.cm as cm  # Used for visualization, not entirely necessary

# Necessary imports for segmentation stage
from scipy import ndimage as img
from utils.rpca_ialm import inexact_augmented_lagrange_multiplier

# Necessary imports for matching stage
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from skimage import measure

# Data structure to store final results
import pandas as pd

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn
seaborn.set()


class FrameQueue:
    """Class for storing, describing, and manipulating frames from a video file
    using two FIFO queues. More or less "collections" deques with additional
    attributes and methods specific to video processing.

    Example:
         Below is a FrameQueue object (size=7), where the last frame read had a
    frame number of 571. Previous frames were pushed deeper into the queue:

    queue:        [i_0][i_1][i_2][i_3][i_4][i_5][i_6] (indexes)
    seg_queue:                   [i_0][i_1][i_2][i_3] (indexes)
    framenumbers: [571][570][569][568][567][566][565] (labelframes in queues)

    The primary queue ("queue") stores original frames, and the secondary queue
    ("seg_queue") stores segmented versions of the frames. As segmentation
    requires contextual information from past/future frames, the center index
    of "queue" (index 3) will correspond to the 0th index in "seg_queue".
    In other words, to generate a segmented version of frame 568,
    frames 565-571 are necessary, and are taken from the primary queue."""

    def __init__(self, args, queue_size=1, desired_fps=False):
        # args.video_dir, args.filename
        # Check validity of filepath
        video_filepath = args.video_dir + args.filename
        if not os.path.isfile(video_filepath):
            raise Exception("[!] Filepath does not point to valid video file.")

        # Open source video file and initialize its immutable attributes
        self.src_filename = args.filename
        self.src_directory = args.video_dir
        self.stream = cv2.VideoCapture("{}/{}".format(self.src_directory,
                                                      self.src_filename))
        if not self.stream.isOpened():
            raise Exception("[!] Video file could not be opened to read"
                            " frames. Check file path.")
        else:
            self.src_fps = self.stream.get(cv2.CAP_PROP_FPS)
            self.src_framecount = int(self.stream.get
                                      (cv2.CAP_PROP_FRAME_COUNT))
            self.src_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.src_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.src_starttime = pd.Timestamp(args.timestamp)

        # Initialize user-defined frame/video attributes
        self.height = self.src_height  # Separate b/c dimensions may change
        self.width = self.src_width    # Separate b/c dimensions may change
        if not desired_fps:
            self.fps = self.src_fps
        else:
            self.fps = desired_fps
        self.delay = round(self.src_fps / self.fps) - 1  # For subsampling vid
        self.frame_to_load_next = 0

        # Generate details for regions of interest in frames
        self.roi, self.crop_region = \
            generate_chimney_regions(args.chimney, 0.25)
        self.roi_mask = self.chimney_roi_segmentation()

        # Initialize primary queue for unaltered frames
        self.queue = collections.deque([], queue_size)
        self.framenumbers = collections.deque([], queue_size)
        self.timestamps = collections.deque([], queue_size)
        self.frames_read = 0

        # Initialize secondary queue for segmented frames
        self.queue_center = int((queue_size - 1) / 2)
        self.seg_queue = collections.deque([], self.queue_center)
        self.seg_properties = collections.deque([], self.queue_center)

    def chimney_roi_segmentation(self):
        """Generate a frame with the chimney's region-of-interest from the
        specified chimney region. Called during __init__ and stored as a
        FrameQueue property."""

        # Read first frame from video file, then reset index back to 0
        success, frame = self.stream.read()
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Apply processing stages to segment roi from cropped frame
        cropped = frame[self.roi[0][1]:self.roi[1][1],
                        self.roi[0][0]:self.roi[1][0]]
        blur = cv2.medianBlur(cv2.medianBlur(cropped, 7), 7)
        a, b, c = cv2.split(blur)
        ret, thr = cv2.threshold(a, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Add roi to empty image of the same size as the frame
        frame_with_thr = np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        frame_with_thr[self.roi[0][1]:self.roi[1][1],
                       self.roi[0][0]:self.roi[1][0]] = thr

        frame_with_thr = self.crop_frame(frame=frame_with_thr)
        frame_with_thr = self.pyramid_down(frame=frame_with_thr, iterations=1)
        _, frame_with_thr = cv2.threshold(frame_with_thr, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return frame_with_thr

    def load_frame_from_video(self):
        """Insert next frame from stream into left side (index 0) of queue."""
        if not self.stream.isOpened():
            raise Exception("[!] Video stream is not open."
                            " Cannot read new frames.")

        # Fetch new frame and update its attributes
        self.framenumbers.appendleft(
            int(self.stream.get(cv2.CAP_PROP_POS_FRAMES)))
        self.timestamps.appendleft(
            self.framenumber_to_timestamp(
                self.stream.get(cv2.CAP_PROP_POS_FRAMES)))
        success, frame = self.stream.read()
        if success:
            self.queue.appendleft(np.array(frame))
            self.frames_read += 1

        # Increments position for next read frame (for skipping frames)
        for i in range(self.delay):
            self.stream.grab()

        return success

    def load_frame_from_file(self, base_save_directory, frame_number,
                             folder_name=None):
        """Insert frame from file into left side (index 0) of queue."""
        # Update frame attributes
        self.framenumbers.appendleft(frame_number)
        timestamp = self.framenumber_to_timestamp(frame_number)
        self.timestamps.appendleft(timestamp)

        # Set appropriate directory
        if not folder_name:
            t = self.timestamps[0].time()
            save_directory = (base_save_directory+"frames/{0:02d}:{1:02d}"
                              .format(t.hour, t.minute))
        else:
            save_directory = base_save_directory+"/"+folder_name

        # Load frame from file ([:11] -> limit precision to get things to work
        # because I increased precision after extracting frames)
        # TODO: Remove this precision limit when frames are re-extracted
        file_paths = glob.glob("{0}/frame{1}_*".format(save_directory,
                                                       frame_number))
        frame = cv2.imread(file_paths[0])
        self.queue.appendleft(np.array(frame))

        if not frame.size == 0:
            success = True
            self.frames_read += 1
        else:
            success = False

        return success

    def save_frame_to_file(self, base_save_directory, frame=None, index=0,
                           scale=1, single_folder=False,
                           base_folder="", frame_folder="frames/",
                           file_prefix="", file_suffix=""):
        """Save an individual frame to an image file. If frame itself is not
        provided, frame will be pulled from frame_queue at specified index."""

        # By default, frames will be saved in a subfolder corresponding to
        # HH:MM formatting. However, a custom subfolder can be chosen instead.
        base_save_directory = base_save_directory+base_folder+frame_folder

        if single_folder:
            save_directory = base_save_directory
        else:
            t = self.timestamps[self.queue_center].time()
            save_directory = (base_save_directory+"{0:02d}:{1:02d}"
                              .format(t.hour, t.minute))

        # Create save directory if it does not already exist
        if not os.path.isdir(save_directory):
            try:
                os.makedirs(save_directory)
                sleep(0.5)  # Sometimes frame 0 won't be saved without a delay
            except OSError:
                print("[!] Creation of the directory {0} failed."
                      .format(save_directory))

        # Extract frame from specified queue object as fallback
        if frame is None:
            frame = self.queue[index]

        # Resize frame for viewing convenience
        if scale is not 1:
            frame = cv2.resize(frame,
                               (round(frame.shape[1]*scale),
                                round(frame.shape[0]*scale)),
                               interpolation=cv2.INTER_AREA)

        # Write frame to image file within save_directory
        try:
            cv2.imwrite("{0}/{1}frame{2}_{3}{4}.jpg"
                        .format(save_directory, file_prefix,
                                self.framenumbers[index],
                                self.timestamps[index].time(),
                                file_suffix),
                        frame)
        except Exception as e:
            print("[!] Image saving failed due to: {0}".format(str(e)))

    def convert_grayscale(self, frame=None, index=0, algorithm="cv2 default"):
        """Convert to grayscale a frame at specified index of FrameQueue"""
        if frame is None:
            frame = self.queue[index]

        if algorithm == "cv2 default":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    def crop_frame(self, frame=None, index=0):
        """Crop frame at specified index of FrameQueue."""
        if frame is None:
            frame = self.queue[index]
        corners = self.crop_region

        try:
            frame = frame[corners[0][1]:corners[1][1],
                          corners[0][0]:corners[1][0]]
        except Exception as e:
            print("[!] Frame cropping failed due to: {0}".format(str(e)))

        # Update frame attributes
        self.height = frame.shape[0]
        self.width = frame.shape[1]

        return frame

    def pyramid_down(self, frame=None, iterations=1, index=0):
        if frame is None:
            frame = self.queue[index]

        for i in range(iterations):
            frame = cv2.pyrDown(frame)

        # Update frame attributes
        self.height = frame.shape[0]
        self.width = frame.shape[1]

        return frame

    def frame_to_column(self, frame=None, index=0):
        """Reshapes an NxM frame into an (N*M)x1 column vector."""
        if frame is None:
            frame = self.queue[index]

        frame = np.squeeze(np.reshape(frame, (self.width*self.height, 1)))

        return frame

    def rpca(self, lmbda, tol, maxiter, darker, index=0):
        """Decompose set of images into corresponding low-rank and sparse
        images. Method expects images to have been reshaped to matrix of
        column vectors.

        Note: frame = lowrank + sparse
                      lowrank = "background" image
                      sparse  = "foreground" errors corrupting
                                the "background" image

        The size of the queue will determine the tradeoff between efficiency
        and accuracy."""

        # np.array alone would give an (#)x(N*M)x1 matrix. Adding transpose
        # and squeeze yields (N*M)x(#). (i.e. a matrix of column vectors)
        matrix = np.squeeze(np.transpose(np.array(self.queue)))

        # Algorithm for the IALM approximation of Robust PCA method.
        lr_columns, s_columns = \
            inexact_augmented_lagrange_multiplier(matrix, lmbda, tol, maxiter)

        # Slice frame from low rank and sparse and reshape back into image
        lr_image = np.reshape(lr_columns[:, index], (self.height, self.width))
        s_image = np.reshape(s_columns[:, index], (self.height, self.width))

        # Bring pixels that are darker than the background into [0, 255] range
        if darker:
            s_image = -1 * s_image  # Darker=negative -> mirror into positive
            np.clip(s_image, 0, 255, s_image)

        return lr_image.astype(dtype=np.uint8), s_image.astype(dtype=np.uint8)

    def segment_frame(self, save_directory, folder_name,
                      params, visual=False):
        """Segment birds from one frame ("index") using information from other
        frames in the FrameQueue object. Store segmented frame in secondary
        queue."""

        # Dictionary for storing segmentation stages (used for visualization)
        seg = {
            "frame": np.reshape(self.queue[self.queue_center],
                                (self.height, self.width))
        }

        # Apply Robust PCA method to isolate regions of motion
        _, seg["RPCA_output"] = self.rpca(params["ialm_lmbda"],
                                          params["ialm_tol"],
                                          params["ialm_maxiter"],
                                          params["ialm_darker"],
                                          index=self.queue_center)

        # Apply thresholding to retain strongest areas and discard the rest
        threshold_str = "thresh_{}".format(params["thr_value"])
        _, seg[threshold_str] = \
            cv2.threshold(list(seg.values())[-1],
                          thresh=params["thr_value"],
                          maxval=255,
                          type=params["thr_type"])

        # Discard areas where 2x2 structuring element will not fit
        seg["grey_opening"] = \
            img.grey_opening(list(seg.values())[-1],
                             size=params["gry_op_SE"]).astype(np.uint8)

        # Segment using connected component labeling
        num_components, labeled_frame = eval(params["seg_func"])

        # Scale labeled image to be visible with uint8 grayscale
        if num_components > 0:
            seg["connected_c_255"] = \
                labeled_frame * int(255 / num_components)
        else:
            seg["connected_c_255"] = labeled_frame

        # Append empty values first if queue is empty
        if self.seg_queue.__len__() is 0:
            self.seg_queue.appendleft(np.zeros((self.height, self.width))
                                      .astype(np.uint8))
            self.seg_properties.appendleft([])

        # Append segmented frame (and information about frame) to queue
        self.seg_queue.appendleft(labeled_frame.astype(np.uint8))
        self.seg_properties.appendleft(measure.regionprops(labeled_frame))

        if visual:
            self.segment_visualization(seg, save_directory, folder_name)

    def segment_visualization(self, seg, save_directory, folder_name):
        # Add roi mask to each stage for visualization.
        for keys, key_values in seg.items():
            seg[keys] = cv2.addWeighted(self.roi_mask, 0.25,
                                        key_values.astype(np.uint8), 0.75, 0)

        # Add filler images if not enough stages to fill gaps
        mod3remainder = len(seg) % 3
        if mod3remainder > 0:
            for i in range(3 - mod3remainder):
                seg["filler_{}".format(i + 1)] = np.zeros((self.height,
                                                           self.width),
                                                          dtype=np.int)

        for keys, key_values in seg.items():
            # Resize frame for visual clarity
            scale = 4  # 400%
            key_values = cv2.resize(key_values.astype(np.uint8),
                                    (round(key_values.shape[1] * scale),
                                    round(key_values.shape[0] * scale)),
                                    interpolation=cv2.INTER_AREA)

            # Apply label to frame
            horizontal_bg = 183 * np.ones(
                shape=(40, key_values.shape[1]),
                dtype=np.uint8)
            seg[keys] = np.vstack((key_values, horizontal_bg))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(seg[keys], keys,
                        (5, seg[keys].shape[0]-10),  # Bottom-left corner
                        font, 1, 0, 2)

        # Concatenate images into Nx3 grid
        rows = [None] * math.ceil((len(seg) / 3))
        sep_h = 64 * np.ones(shape=(list(seg.values())[0].shape[0], 2),
                             dtype=np.uint8)
        for i in range(len(rows)):
            # Concatenate 3 images into 1x3 row
            rows[i] = np.hstack((list(seg.values())[0 + (i*3)], sep_h,
                                 list(seg.values())[1 + (i*3)], sep_h,
                                 list(seg.values())[2 + (i*3)]))
            # If more than one row, stack rows together
            if i > 0:
                sep_v = 64 * np.ones(shape=(2, rows[0].shape[1]),
                                     dtype=np.uint8)
                rows[0] = np.vstack((rows[0], sep_v,
                                     rows[i])).astype(np.uint8)

        # Save to file
        self.save_frame_to_file(save_directory,
                                frame=rows[0],
                                index=self.queue_center,
                                base_folder=folder_name,
                                frame_folder="visualizations/segmentation/")

    def match_segments(self, save_directory, folder_name,
                       params, visual=False):
        """Analyze a pair of segmented frames and return conclusions about
        which segments match between frames."""
        
        # Assign name to commonly used properties
        count = len(self.seg_properties[0])
        count_prev = len(self.seg_properties[1])
        count_total = count + count_prev

        # Initialize coordinates and counts
        matches = []
        coords = [[(0, 0) for col in range(2)] for row in range(count_total)]
        counts = {
            "TMSTAMP": self.timestamps[self.queue_center],
            "FRM_NUM": self.framenumbers[self.queue_center],
            "SEGMNTS": count,
            "MATCHES": 0,
            "ENT_CHM": 0,
            "ENT_FRM": 0,
            "ENT_AMB": 0,
            "EXT_CHM": 0,
            "EXT_FRM": 0,
            "EXT_AMB": 0,
            "OUTLIER": 0,
            "SEG_ERR": 0
        }
        frame_err = 0
        frame_prev_err = 0

        # Compute and analyze match pairs only if segments exist
        if count_total > 0:
            # Initialize likelihood matrix
            likeilihood_matrix = np.zeros((count_total, count_total))

            # Matrix values: likelihood of segments being a match
            for seg_prev in self.seg_properties[1]:
                for seg in self.seg_properties[0]:
                    # Convert segment labels to likelihood matrix indices
                    index_v = (seg_prev.label - 1)
                    index_h = (count_prev + seg.label - 1)

                    # Likeilihoods as a function of distance between segments
                    dist = distance.euclidean(seg_prev.centroid,
                                              seg.centroid)
                    # Map distance values using a Gaussian curve
                    likeilihood_matrix[index_v, index_h] \
                        = eval(params["ap_func_match"])

            # Matrix values: likelihood of segments appearing/disappearing
            for i in range(count_total):
                # Compute closest distance from segment to edge of frame
                if i < count_prev:
                    point = self.seg_properties[1][i].centroid
                if count_prev <= i < (count + count_prev):
                    point = self.seg_properties[0][i - count_prev].centroid
                edge_distance = min([point[0], point[1],
                                     self.height - point[0],
                                     self.width - point[1]])

                # Map distance values using an Exponential curve
                likeilihood_matrix[i, i] = eval(params["ap_func_notmatch"])

            # Convert likelihood matrix into cost matrix
            cost_matrix = -1*likeilihood_matrix
            cost_matrix -= cost_matrix.min()

            # Apply Hungarian/Munkres algorithm to find optimal matches
            _, matches = linear_sum_assignment(cost_matrix)

            # Convert matches (pairs of indices) into pairs of coordinates
            for i in range(count_total):
                j = matches[i]
                # Index condition if two segments are matching
                if (i < j) and (i < count_prev):
                    coords[i][0] = self.seg_properties[1][i].centroid
                    coords[i][1] = self.seg_properties[0][j-count_prev].centroid
                # Index condition for when a previous segment has disappeared
                elif (i == j) and (i < count_prev):
                    coords[i][0] = self.seg_properties[1][i].centroid
                # Index condition for when a new segment has appeared
                elif (i == j) and (i >= count_prev):
                    coords[i][1] = self.seg_properties[0][j-count_prev].centroid

            # For each pair of coordinates, classify as certain behaviors
            for coord_pair in coords:
                if coord_pair[0] == (0, 0) and coord_pair[1] == (0, 0):
                    pass
                # If this condition is met, pair means a segment appeared
                elif coord_pair[0] == (0, 0):
                    edge_distance = min(coord_pair[1][0], coord_pair[1][1],
                                        self.width - coord_pair[1][1])
                    roi_value = self.roi_mask[int(coord_pair[1][0])] \
                                             [int(coord_pair[1][1])]

                    if edge_distance <= 10:
                        counts["ENT_FRM"] += 1
                    elif roi_value == 255:
                        counts["ENT_CHM"] += 1
                    else:
                        counts["SEG_ERR"] += 1
                        frame_err += 1
                # If this condition is met, pair means a segment disappeared
                elif coord_pair[1] == (0, 0):
                    edge_distance = min(coord_pair[0][0], coord_pair[0][1],
                                        self.width - coord_pair[0][1])
                    roi_value = self.roi_mask[int(coord_pair[0][0])] \
                                             [int(coord_pair[0][1])]

                    if edge_distance <= 10:
                        counts["EXT_FRM"] += 1
                    elif roi_value == 255:
                        counts["EXT_CHM"] += 1
                    else:
                        counts["SEG_ERR"] += 1
                        frame_prev_err += 1
                # Otherwise, a match was made
                else:
                    counts["MATCHES"] += 1

        # Create visualization of segment matches if requested
        if visual:
            self.match_visualization(count_prev, count_total,
                                     matches, counts,
                                     frame_prev_err, frame_err,
                                     save_directory, folder_name)

        return counts

    def match_visualization(self, count_prev, count_total,
                            matches, counts,
                            frame_prev_err, frame_err,
                            save_directory, folder_name):
        frame = np.copy(self.seg_queue[0])
        frame_prev = np.copy(self.seg_queue[1])

        # Colormappings for tab20 colormap in colormap
        colors = [14, 40, 118, 144, 170, 222, 248,
                  1, 27, 105, 131, 157, 209, 235]  # non-G/R colors
        appeared_color = 53  # Green
        disappeared_color = 79  # Red

        # Color matching between frames (grayscale, colormap comes later)
        frame_prev[frame_prev == 0] = 196
        frame[frame == 0] = 196
        color_index = 0
        for i in range(count_total):
            j = matches[i]
            # Index condition if two segments are matching
            if (i < j) and (i < count_prev):
                frame_prev[frame_prev == (i + 1)] = colors[color_index % 14]
                frame[frame == (j + 1 - count_prev)] = colors[color_index % 14]
                color_index += 1
            # Index condition for when a previous segment has disappeared
            elif (i == j) and (i < count_prev):
                frame_prev[frame_prev == (i + 1)] = disappeared_color
            # Index condition for when a new segment has appeared
            elif (i == j) and (i >= count_prev):
                frame[frame == (j + 1 - count_prev)] = appeared_color

        # Resize frames for visual convenience
        scale = 4
        frame = cv2.resize(frame,
                           (round(frame.shape[1] * scale),
                            round(frame.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
        frame_prev = cv2.resize(frame_prev,
                                (round(frame_prev.shape[1] * scale),
                                 round(frame_prev.shape[0] * scale)),
                                interpolation=cv2.INTER_AREA)

        # Combine both frames into single image
        separator_v = 183 * np.ones(shape=(self.height * scale, 1),
                                    dtype=np.uint8)
        match_comparison = np.hstack((frame_prev, separator_v, frame))

        # Adding horizontal bar to display frame information
        horizontal_bg = 183 * np.ones(shape=(50, match_comparison.shape[1]),
                                      dtype=np.uint8)
        match_comparison = np.vstack((match_comparison, horizontal_bg))

        # Write text on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(match_comparison,
                    'Frame{0} - Ext, edge: {1} | Ext, chimn: {2} | '
                    'False positive: {3}    '
                    'Frame{4} - Ent, edge: {5} | Ent, chimn: {6} | '
                    'False positive: {7}'.format(counts["FRM_NUM"] - 1,
                                                 counts["EXT_FRM"],
                                                 counts["EXT_CHM"],
                                                 frame_prev_err,
                                                 counts["FRM_NUM"],
                                                 counts["ENT_FRM"],
                                                 counts["ENT_CHM"],
                                                 frame_err),
                    (10, (self.height*scale+50)-10), font, 1, 196, 2)

        # Combine two ROI masks into single image.
        roi_mask = cv2.resize(self.roi_mask,
                           (round(self.roi_mask.shape[1] * scale),
                            round(self.roi_mask.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
        separator_v = np.zeros(shape=(self.height*scale, 1), dtype=np.uint8)
        roi_masks = np.hstack((roi_mask, separator_v, roi_mask))

        # Adding horizontal bar to display frame information
        bar = np.zeros(shape=(50, roi_masks.shape[1]), dtype=np.uint8)
        roi_masks = np.vstack((roi_masks, bar))
        roi_stacked = np.stack((roi_masks,) * 3, axis=-1).astype(np.uint8)

        # Apply color mapping
        match_comparison_color = cm.apply_custom_colormap(match_comparison,
                                                          cmap="tab20")

        match_comparison_color = \
            cv2.addWeighted(roi_stacked, 0.10,
                            match_comparison_color, 0.90, 0)


        # Save image to folder
        self.save_frame_to_file(save_directory,
                                frame=match_comparison_color,
                                index=self.queue_center,
                                base_folder=folder_name,
                                frame_folder="visualizations/matching/",
                                scale=1)

    def framenumber_to_timestamp(self, frame_number):
        """Helper function to convert an amount of frames into a timestamp."""
        # cv2's VideoCapture class provides a frame count property
        # (cv2.CAP_PROP_FRAME_COUNT) but not a duration property. However, as it
        # can be easier to think of video in terms of hours, minutes, and seconds,
        # it's helpful to be able to convert back and forth.
        total_s = frame_number / self.fps
        timestamp = self.src_starttime + pd.Timedelta(total_s, 's')

        return timestamp

    def timestamp_to_framenumber(self, timestamp):
        """Helper function to convert timestamp into an amount of frames."""
        # cv2's VideoCapture class provides a frame count property
        # (cv2.CAP_PROP_FRAME_COUNT) but not a duration property. However, as it
        # can be easier to think of video in terms of hours, minutes, and seconds,
        # it's helpful to be able to convert back and forth.
        t = timestamp.time()
        total_s = (t.hour * 60 * 60 +
                   t.minute * 60 +
                   t.second +
                   t.microsecond / 1e6)
        frame_number = int(round(total_s * self.fps))

        return frame_number


def process_extracted_frames(args, params):
    """Function which uses class methods to analyse a sequence of previously
    extracted frames, and determine bird counts for that sequence."""

    # FrameQueue object (class for caching frames and applying
    #                    image processing to cached frames)
    frame_queue = FrameQueue(args, queue_size=params["queue_size"])
    frame_queue.stream.release()  # VideoCapture not needed for frame reuse
    frame_queue.frame_to_load_next = args.load[0]
    num_frames_to_analyse = args.load[1] - args.load[0]

    # Empty list. Will be filled with a dictionary of counts for each frame.
    # Then, list of dictionaries will be converted to pandas DataFrame.
    count_estimate = []

    print("[========================================================]")
    print("[*] Analysing frames... (This may take a while!)")

    # The number of frames to read has an additional amount added,
    # "frame_queue.queue_center", because a cache of frames is needed to
    # segment a frame. (Sequential context for motion estimation.)
    # See pv.FrameQueue's __init__() docstring for more information.
    while frame_queue.frames_read < num_frames_to_analyse:

        # Load frame into index 0 and apply preprocessing
        frame_queue.load_frame_from_file(args.default_dir,
                                         frame_queue.frame_to_load_next)
        frame_queue.queue[0] = \
            frame_queue.convert_grayscale(algorithm=params["gs_algorithm"])
        frame_queue.queue[0] = frame_queue.crop_frame()
        frame_queue.queue[0] = frame_queue.pyramid_down(iterations=1)
        frame_queue.queue[0] = frame_queue.frame_to_column()

        # Proceed only when enough frames are stored for motion estimation
        if frame_queue.frames_read > frame_queue.queue_center:
            frame_queue.segment_frame(args.default_dir,
                                      args.custom_dir,
                                      params,
                                      visual=args.visual)
            match_counts = frame_queue.match_segments(args.default_dir,
                                                      args.custom_dir,
                                                      params,
                                                      visual=args.visual)
            count_estimate.append(match_counts)

        # Status updates
        if frame_queue.frames_read % 25 == 0:
            print("[-] {0}/{1} frames processed."
                  .format(frame_queue.frames_read, num_frames_to_analyse))

        # Delay = 0 if fps == src_fps, delay > 0 if fps < src_fps
        frame_queue.frame_to_load_next += (1 + frame_queue.delay)

    print("[-] Analysis complete. {} total frames used in counting."
          .format(frame_queue.frames_read - frame_queue.queue_center))

    # Specifying exactly which columns should be used for DataFrame
    columns = ["TMSTAMP", "FRM_NUM",
               "SEGMNTS", "MATCHES",
               "ENT_CHM", "ENT_FRM", "ENT_AMB",
               "EXT_CHM", "EXT_FRM", "EXT_AMB",
               "OUTLIER"]

    # Convert dictionary of lists into DataFrame
    df_estimation = pd.DataFrame(count_estimate, columns=columns)
    df_estimation.set_index("TMSTAMP", inplace=True)
    df_estimation.index = pd.to_datetime(df_estimation.index)

    return df_estimation


def extract_frames(args, queue_size=1, save_directory=None):
    """Function which uses class methods to extract individual frames
     (one at a time) from a video file. Saves each frame to image files for
     future reuse."""

    # Default save directory chosen to be identical to filename
    if not save_directory:
        save_directory = args.default_dir

    print("[========================================================]")
    print("[*] Reading frames... (This may take a while!)")

    frame_queue = FrameQueue(args, queue_size)
    while frame_queue.frames_read < frame_queue.src_framecount:
        # Attempt to read frame from video into queue object
        success = frame_queue.load_frame_from_video()

        # Save frame to file if read correctly
        if success:
            frame_queue.save_frame_to_file(save_directory)
        else:
            raise Exception("read_frame() failed before expected end of file.")

        # Status updates
        if frame_queue.frames_read % 1000 == 0:
            print("[-] {}/{} frames successfully processed."
                  .format(frame_queue.frames_read, frame_queue.src_framecount))
    frame_queue.stream.release()

    print("[========================================================]")
    print("[-] Extraction complete. {} total frames extracted."
          .format(frame_queue.frames_read))


def generate_chimney_regions(bottom_corners, alpha):
    width = bottom_corners[1][0] - bottom_corners[0][0]
    height = round(alpha*width)

    # Outside coordinates from provided corners
    left = min(bottom_corners[0][0], bottom_corners[1][0])
    right = max(bottom_corners[0][0], bottom_corners[1][0])
    top = min(bottom_corners[0][1], bottom_corners[1][1])
    bottom = max(bottom_corners[0][1], bottom_corners[1][1])

    crop_region = [(left - height, top - 3*height),
                   (right + height, bottom + height)]
    roi_region = [(int(left - 0.05*width), int(bottom - height)),
                  (int(left + 1.05*width), int(bottom))]

    return roi_region, crop_region



