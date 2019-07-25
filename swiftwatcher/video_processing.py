# Stdlib imports
import os
import glob
import collections
import math
from time import sleep

# Imports used in numerous stages
import cv2
import numpy as np

# Necessary imports for segmentation stage
from scipy import ndimage as img
from utils.rpca_ialm import inexact_augmented_lagrange_multiplier

# Necessary imports for matching stage
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from skimage import measure
import utils.cm as cm
import pandas as pd

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

class FrameQueue:
    """Class for storing, describing, and manipulating frames from a video file
    using two FIFO queues. More or less collections.deques with additional
    attributes and methods specific to processing swift video files.

    Example:
         Below is a FrameQueue object (size=7), where the most recently read
         frame had a framenumber of 571. Previously read frames were pushed
         deeper into the queue:

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

        def assign_file_properties():
            # Check validity of filepath
            video_filepath = args.video_dir + args.filename
            if not os.path.isfile(video_filepath):
                raise Exception(
                    "[!] Filepath does not point to valid video file.")

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
                self.src_height = int(
                    self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.src_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.src_starttime = pd.Timestamp(args.timestamp)

        def assign_processing_properties():
            # Separate from src in case of cropping
            self.height = self.src_height
            self.width = self.src_width

            if not desired_fps:
                self.fps = self.src_fps
            else:
                self.fps = desired_fps
            self.delay = round(
                self.src_fps / self.fps) - 1  # For subsampling vid

            if "load" not in args:
                args.load = [0, -1]

            self.frame_to_load_next = args.load[0]
            if args.load[1] == -1:
                self.total_frames \
                    = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                self.total_frames = args.load[1] - args.load[0] + 1

            # Initialize "disappeared segment" event tracking list
            self.event_list = []

        def assign_queue_properties():
            # Initialize primary queue for unaltered frames
            self.queue = collections.deque([], queue_size)
            self.framenumbers = collections.deque([], queue_size)
            self.timestamps = collections.deque([], queue_size)
            self.frames_read = 0
            self.frames_processed = 0

            # Initialize secondary queue for segmented frames
            self.queue_center = int((queue_size - 1) / 2)
            self.seg_queue = collections.deque([], self.queue_center)
            self.seg_properties = collections.deque([], self.queue_center)

        def generate_chimney_regions(alpha):
            """Generate rectangular regions (represented as top-left corner and
            bottom-right corner) from two provided points ("bottom_corners").

            The two points provided are of the two edges of the chimney:

                         (x1, y1) *-----------------* (x2, y2)
                                  |                 |
                                  |  chimney stack  |
                                  |                 |                       """

            # Recording the outer-most coordinates from the two provided points
            # because they may not be parallel.
            bottom_corners = args.chimney
            left = min(bottom_corners[0][0], bottom_corners[1][0])
            right = max(bottom_corners[0][0], bottom_corners[1][0])
            top = min(bottom_corners[0][1], bottom_corners[1][1])
            bottom = max(bottom_corners[0][1], bottom_corners[1][1])

            width = right - left
            height = round(alpha * width)  # Fixed height/width ratio
            self.crop_region = [(left - int(0.5*height),
                                 top - 2*height),
                                (right + int(0.5*height),
                                 bottom + int(0.5*height))]
            self.nn_region = [(left - int(0.5*height),
                                 top - height),
                                (right + int(0.5*height),
                                 bottom)]
            # NOTE: I think he "crop_region" is too large -- I believe a
            # smaller region would produce similar results. For example:
            # crop_region = [(left, top - height), (right, bottom + height)]
            #
            # Because choosing a different crop would require recomputing the
            # RPCA step (which is a bottleneck for time), this won't be touched
            # for a while.

            self.roi_region = [(int(left + 0.025 * width),
                                int(bottom - height)),
                               (int(right - 0.025 * width),
                                int(bottom))]

        def generate_roi_mask():
            """Generate a mask with the chimney's region-of-interest from the
            specified chimney region. Mask will be the same dimensions as
            "crop_region". Func called during __init__ and stored as a property."""

            # Read first frame from video file, then reset index back to 0
            success, frame = self.stream.read()
            self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Crop frame, then blur and threshold the B channel to produce mask
            cropped = frame[self.roi_region[0][1]:self.roi_region[1][1],
                      self.roi_region[0][0]:self.roi_region[1][0]]
            blur = cv2.medianBlur(cv2.medianBlur(cropped, 9), 9)
            a, b, c = cv2.split(blur)
            ret, thr = cv2.threshold(a, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thr = cv2.Canny(thr, 0, 256)
            thr = cv2.dilate(thr, kernel=np.ones((20, 1), np.uint8),
                             anchor=(0, 0))

            # NOTE: I've chosen a rectangular ROI here but I think this isn't the
            # right approach to take. I think the ROI should be shaped like the
            # edge of the chimney to more accurately represent regions where
            # swifts exist right before entering the chimney.
            #    ______________________              ________________
            #   |X   ______________   X|           ,' ______________ ',
            #   |  ,'              ',  |          / ,'              ', \
            #   | /                  \ |         | /    potential     \ |
            #   |/      current       \|         |/    alternative     \|
            #
            # X's represent area in the current ROI where false positives occur.

            # Add roi to empty image of the same size as the frame
            frame_with_thr = np.zeros_like(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            frame_with_thr[self.roi_region[0][1]:self.roi_region[1][1],
                           self.roi_region[0][0]:self.roi_region[1][0]] = thr

            # Crop/resample mask (identical preprocessing to the actual frames).
            frame_with_thr = self.preprocess_frame(frame_with_thr)
            _, self.roi_mask = cv2.threshold(frame_with_thr, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # NOTE: This is a hacky way to do this -- there's probably a nicer way
            # to ensure that the ROI gets the same preprocessing as the frames
            # themselves. I could probably wrap the preprocessing stages in a
            # single function rather than having them separate. Time concerns, etc

        assign_file_properties()
        assign_processing_properties()
        assign_queue_properties()
        generate_chimney_regions(alpha=0.25)
        generate_roi_mask()

        if "extract" in args:
            if not args.extract:
                self.stream.release()

    def load_frame(self, load_directory=None, empty=False):

        def fn_to_ts(frame_number):
            """Helper function to convert an amount of frames into a timestamp."""
            total_s = frame_number / self.fps
            timestamp = self.src_starttime + pd.Timedelta(total_s, 's')

            return timestamp

        def ts_to_fn(timestamp):
            """Helper function to convert timestamp into an amount of frames."""
            t = timestamp.time()
            total_s = (t.hour * 60 * 60 +
                       t.minute * 60 +
                       t.second +
                       t.microsecond / 1e6)
            frame_number = int(round(total_s * self.fps))

            return frame_number

        def load_frame_from_video():
            """Insert next frame from stream into index 0 of queue."""
            nonlocal new_timestamp, new_framenumber, new_frame, success

            # Fetch new frame and update its attributes
            new_framenumber = int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))
            new_timestamp = fn_to_ts(self.stream.get(cv2.CAP_PROP_POS_FRAMES))
            success, frame = self.stream.read()
            if success:
                new_frame = np.array(frame)
                self.frames_read += 1

            # Increments position for next read frame (for skipping frames)
            for i in range(self.delay):
                self.stream.grab()

        def load_frame_from_file():
            """Insert frame from file into index 0 of queue."""
            nonlocal new_timestamp, new_framenumber, new_frame, success

            new_framenumber = self.frame_to_load_next
            new_timestamp = fn_to_ts(new_framenumber)

            t = new_timestamp.time()
            directory = (load_directory +
                         "frames/{0:02d}:{1:02d}".format(t.hour, t.minute))

            file_paths = glob.glob("{0}/frame{1}_*".format(directory,
                                                           new_framenumber))
            new_frame = np.array(cv2.imread(file_paths[0]))

            if not new_frame.size == 0:
                success = True
                self.frames_read += 1

            self.frame_to_load_next += (1+self.delay)

        new_timestamp = ""
        new_framenumber = ""
        new_frame = ""
        success = False

        if load_directory:
            load_frame_from_file()
        elif empty:
            pass
        else:
            load_frame_from_video()

        self.timestamps.appendleft(new_timestamp)
        self.framenumbers.appendleft(new_framenumber)
        self.queue.appendleft(new_frame)

        return success

    def save_frame(self, base_save_directory, frame=None, index=0,
                   scale=1, single_folder=False,
                   base_folder="", frame_folder="frames/",
                   file_prefix="", file_suffix=""):
        """Save an individual frame to an image file. If frame itself is not
        provided, frame will be pulled from frame queue at specified index."""

        # By default, frames will be saved in a subfolder corresponding to
        # HH:MM formatting. However, a custom subfolder can be chosen instead.
        base_save_directory = base_save_directory+base_folder+frame_folder

        if single_folder:
            save_directory = base_save_directory
        else:
            t = self.timestamps[-1].time()
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
        cv2.imwrite("{0}/{1}frame{2}_{3}{4}.jpg"
                    .format(save_directory, file_prefix,
                            self.framenumbers[index],
                            self.timestamps[index].time(),
                            file_suffix),
                    frame)

    def preprocess_frame(self, frame=None, index=0, iterations=1):

        def convert_grayscale():
            """Convert to grayscale a frame at specified index of FrameQueue"""

            nonlocal frame

            # OpenCV default, may use other methods in future (single HSV channel?)
            if len(frame.shape) is 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        def crop_frame():
            """Crop frame at specified index of FrameQueue."""

            nonlocal frame
            corners = self.crop_region

            try:
                frame = frame[corners[0][1]:corners[1][1],
                              corners[0][0]:corners[1][0]]
            except Exception as e:
                print("[!] Frame cropping failed due to: {0}".format(str(e)))

            # Update frame attributes
            self.height = frame.shape[0]
            self.width = frame.shape[1]

        def pyramid_down():
            nonlocal frame

            for i in range(iterations):
                frame = cv2.pyrDown(frame)

            # Update frame attributes
            self.height = frame.shape[0]
            self.width = frame.shape[1]

        if frame is None:
            passed_frame = False
        else:
            passed_frame = True

        if not passed_frame:
            frame = self.queue[index]

        convert_grayscale()
        crop_frame()
        pyramid_down()

        if not passed_frame:
            self.queue[index] = frame
        else:
            return frame

    def segment_frame(self, args, params):
        """Segment birds from one frame ("index") using information from other
        frames in the FrameQueue object. Store segmented frame in secondary
        queue."""

        def rpca(lmbda, tol, maxiter, darker, index=None):
            """Decompose set of images into corresponding low-rank and sparse
            images. Method expects images to have been reshaped to matrix of
            column vectors.

            Note: frame = lowrank + sparse
                          lowrank = "background" image
                          sparse  = "foreground" errors corrupting
                                    the "background" image

            The size of the queue will determine the tradeoff between efficiency
            and accuracy."""

            img_matrix = np.array(self.queue)
            if index:
                img_matrix = img_matrix[:index, :, :]

            col_matrix = np.transpose(img_matrix.reshape(img_matrix.shape[0],
                                                         img_matrix.shape[1] *
                                                         img_matrix.shape[2]))

            # Algorithm for the IALM approximation of Robust PCA method.
            lr_columns, s_columns = \
                inexact_augmented_lagrange_multiplier(col_matrix,
                                                      lmbda, tol, maxiter)

            # Bring pixels that are darker than background to [0, 255] range
            if darker:
                s_columns = np.negative(s_columns)
                s_columns = np.clip(s_columns, 0, 255).astype(np.uint8)

            # Reshape columns back into image dimensions
            for i in range(img_matrix.shape[0]):
                self.queue[i] = np.reshape(s_columns[:, i],
                                           (self.height, self.width))

        def edge_based_otsu(image):
            # smoothed_image = cv2.medianBlur(image, 5)
            smoothed_image = cv2.bilateralFilter(image, d=7, sigmaColor=15,
                                                 sigmaSpace=1)
            edge_image = cv2.Canny(smoothed_image, 100, 200)
            mask = cv2.dilate(edge_image, np.ones((2, 2), np.uint8),
                              iterations=2).astype(np.int)

            # Change mask so all 0 values will make original image negative
            # (Therefore excluding those values from otsu's thresholding)
            mask[mask == 0] = (-256)
            mask[mask == 255] = 0
            masked_image = np.add(mask, smoothed_image).astype(np.uint8)

            # Get a threshold value from only the edge (+ edge-adjacent) values
            ret, _ = cv2.threshold(masked_image, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            _, thresholded_image = cv2.threshold(smoothed_image,
                                                 thresh=ret, maxval=255,
                                                 type=cv2.THRESH_TOZERO)

            return thresholded_image

        def segment_visualization():
            # Add roi mask to each stage for visualization.
            for keys, key_values in seg.items():
                seg[keys] = cv2.addWeighted(self.roi_mask, 0.05,
                                            key_values.astype(np.uint8), 0.95,
                                            0)

            # Add filler images if not enough stages to fill Nx3 grid
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
                            (5, seg[keys].shape[0] - 10),  # Bottom-left corner
                            font, 1, 0, 2)

            # Concatenate images into Nx3 grid
            rows = [None] * math.ceil((len(seg) / 3))
            sep_h = 64 * np.ones(shape=(list(seg.values())[0].shape[0], 2),
                                 dtype=np.uint8)
            for i in range(len(rows)):
                # Concatenate 3 images into 1x3 row
                rows[i] = np.hstack((list(seg.values())[0 + (i * 3)], sep_h,
                                     list(seg.values())[1 + (i * 3)], sep_h,
                                     list(seg.values())[2 + (i * 3)]))
                # If more than one row, stack rows together
                if i > 0:
                    sep_v = 64 * np.ones(shape=(2, rows[0].shape[1]),
                                         dtype=np.uint8)
                    rows[0] = np.vstack((rows[0], sep_v,
                                         rows[i])).astype(np.uint8)

            # Save to file
            self.save_frame(save_directory,
                            frame=rows[0],
                            index=-1,
                            base_folder=folder_name,
                            frame_folder="visualizations/segmentation/")

        if "visual" not in args:
            save_directory = None
            folder_name = None
            visual = False
        else:
            save_directory = args.default_dir
            folder_name = args.custom_dir
            visual = args.visual

        # Dictionary for storing segmentation stages (used for visualization)
        seg = {
            # "frame": np.reshape(self.queue[-1],
            #                     (self.height, self.width))
        }

        # Apply Robust PCA method to isolate regions of motion
        if self.frames_read % params["queue_size"] == 0:
            rpca(params["ialm_lmbda"], params["ialm_tol"],
                 params["ialm_maxiter"], params["ialm_darker"])
        if self.frames_read == self.total_frames:
            if self.frames_read-self.frames_processed == params["queue_size"]:
                rem = self.total_frames % params["queue_size"]
                rpca(params["ialm_lmbda"], params["ialm_tol"],
                     params["ialm_maxiter"], params["ialm_darker"], index=rem)
        seg["RPCA_output"] = self.queue[-1]
        _, seg["RPCA_hardthr"] = cv2.threshold(list(seg.values())[-1],
                                               thresh=15,
                                               maxval=255,
                                               type=cv2.THRESH_TOZERO)
        seg["RPCA_otsuthr"] = edge_based_otsu(list(seg.values())[-1])
        seg["RPCA_opened"] = img.grey_opening(list(seg.values())[-1],
                                              size=(3, 3)).astype(np.uint8)


        # plt.cla()
        # plt.hist(seg["RPCA_masked"].ravel(), 256, [0, 256])
        # fig.canvas.draw()
        # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # data_gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        # seg["histogram"] = cv2.resize(data_gray, (seg["RPCA_masked"].shape[1],
        #                                           seg["RPCA_masked"].shape[0]))



        # # Apply thresholding to retain strongest areas and discard the rest
        # threshold_str = "thresh_{}".format(params["thr_value"])
        # _, seg[threshold_str] = \
        #     cv2.threshold(list(seg.values())[-1],
        #                   thresh=params["thr_value"],
        #                   maxval=255,
        #                   type=params["thr_type"])
        #
        # # Discard areas where 2x2 structuring element will not fit
        # for i in range(len(params["grey_op_SE"])):
        #     seg["grey_opening{}".format(i+1)] = \
        #         img.grey_opening(list(seg.values())[-1],
        #                          size=params["grey_op_SE"][i]).astype(np.uint8)
        #
        # Segment using connected component labeling
        num_components, labeled_frame = \
            cv2.connectedComponents(list(seg.values())[-1], connectivity=4)

        # Scale labeled image to be visible with uint8 grayscale
        if num_components > 0:
            seg["connected_c_255"] = \
                labeled_frame * int(255 / num_components)
        else:
            seg["connected_c_255"] = labeled_frame

        # Connected component labeling of non-processed image for demonstration
        num_componentsd, labeled_framed = \
            cv2.connectedComponents(seg["RPCA_output"], connectivity=4)

        # Scale labeled image to be visible with uint8 grayscale
        if num_componentsd > 0:
            seg["connected_c_255d"] = \
                labeled_framed * int(255 / num_componentsd)
        else:
            seg["connected_c_255d"] = labeled_framed
        #
        # # Append empty values first if queue is empty
        # if self.seg_queue.__len__() is 0:
        #     self.seg_queue.appendleft(np.zeros((self.height, self.width))
        #                               .astype(np.uint8))
        #     self.seg_properties.appendleft([])
        #
        # # Append segmented frame (and information about frame) to queue
        # self.seg_queue.appendleft(labeled_frame.astype(np.uint8))
        # self.seg_properties.appendleft(measure.regionprops(labeled_frame))

        self.frames_processed += 1

        if visual:
            segment_visualization()

    def match_segments(self, args, params):
        """Analyze a pair of segmented frames and return conclusions about
        which segments match between frames.

        Example: Matching 4 segments in the current frame to 3 segments in the
                 previous frame.

                                 current
                            frame's segments

                               0  1  2  3
        previous   0 [D][ ][ ][!][!][!][!]
         frame's   1 [ ][D][ ][!][!][!][!]
        segments   2 [ ][ ][D][!][!][!][!]
                     [ ][ ][ ][A][ ][ ][ ]
                     [ ][ ][ ][ ][A][ ][ ]
                     [ ][ ][ ][ ][ ][A][ ]
                     [ ][ ][ ][ ][ ][ ][A]

        -The "!" values in the matrix are likelihoods of two segments matching.
        -The "D" and "A" values in the matrix are likelihoods of that segment
         having no match. (Disappeared and appeared respectively.)
        -The " " values are irrelevant spaces.

        By using the Hungarian algorithm, this matrix is solved to give a
        single outcome (match, no match) to each segment in both frames. The
        total sum of likelihoods will be the maximum across all possible
        combinations of matches, as per the Assignment Problem."""

        def match_visualization():
            """Create visualizations from matching results and segmented frames."""
            # Colormappings for tab20 colormap.
            # See: https://matplotlib.org/examples/color/colormaps_reference.html
            colors = [14, 40, 118, 144, 170, 222, 248,
                      # non-G/R colors (pastel)
                      1, 27, 105, 131, 157, 209,
                      235]  # non-G/R colors (normal)
            appeared_color = 53  # Green
            fp_appeared_color = 66  # Pastel green
            disappeared_color = 79  # Red
            fp_disappeared_color = 82  # Pastel red
            background_color = 196  # Light grey

            # Replacing connected component labeling output (0 for background,
            # 1, 2, 3... for segments) with matched colormapping. Grayscale values
            # (0-255) correspond to colors in tab20 qualitative colormap.
            frame = np.copy(self.seg_queue[0])
            frame_prev = np.copy(self.seg_queue[1])
            frame_prev[frame_prev == 0] = background_color
            frame[frame == 0] = background_color
            color_index = 0
            for i in range(count_total):
                j = seg_matches[i]
                # Index condition if two segments are matching
                if (i < j) and (i < count_prev):
                    frame_prev[frame_prev == (i + 1)] = colors[
                        color_index % 14]
                    frame[frame == (j + 1 - count_prev)] = colors[
                        color_index % 14]
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

            # Combine two ROI masks into single image.
            roi_mask = cv2.resize(self.roi_mask,
                                  (round(self.roi_mask.shape[1] * scale),
                                   round(self.roi_mask.shape[0] * scale)),
                                  interpolation=cv2.INTER_AREA)
            separator_v = np.zeros(shape=(self.height * scale, 1),
                                   dtype=np.uint8)
            roi_masks = np.hstack((roi_mask, separator_v, roi_mask))
            roi_stacked = np.stack((roi_masks,) * 3, axis=-1).astype(np.uint8)

            # Apply color mapping, then apply mask to colormapped image
            match_comparison_color = cm.apply_custom_colormap(match_comparison,
                                                              cmap="tab20")
            match_comparison_color = \
                cv2.addWeighted(roi_stacked, 0.10,
                                match_comparison_color, 0.90, 0)

            # Save completed visualization to folder
            self.save_frame(save_directory,
                                    frame=match_comparison_color,
                                    index=-1,
                                    base_folder=folder_name,
                                    frame_folder="visualizations/matching/",
                                    scale=1)

        if "visual" not in args:
            save_directory = None
            folder_name = None
            visual = False
        else:
            save_directory = args.default_dir
            folder_name = args.custom_dir
            visual = args.visual

        # Assign names to commonly used properties
        count_curr = len(self.seg_properties[0])
        count_prev = len(self.seg_properties[1])
        count_total = count_curr + count_prev

        if count_total > 0:
            # Initialize likelihood matrix as (N+M)x(N+M)
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
                    # NOTE: This function was scrapped together in June. Needs
                    # to be chosen more methodically if used for paper.
                    likeilihood_matrix[index_v, index_h] = \
                        math.exp(-1 * (((dist - 5) ** 2) / 40))

            # Matrix values: likelihood of segments having no match
            for i in range(count_total):
                # Compute closest distance from segment to edge of frame
                if i < count_prev:
                    point = self.seg_properties[1][i].centroid
                if count_prev <= i < (count_curr + count_prev):
                    point = self.seg_properties[0][i - count_prev].centroid
                edge_distance = min([point[0], point[1],
                                     self.height - point[0],
                                     self.width - point[1]])

                # Map distance values using an Exponential curve
                # NOTE: This function was scrapped together in June. Needs to
                # be chosen more methodically if used for paper.
                likeilihood_matrix[i, i] = \
                    (1 / 8) * math.exp(-edge_distance / 10)

            # Convert likelihood matrix into cost matrix
            # This is necessary because of scipy's default implementation
            cost_matrix = -1*likeilihood_matrix
            cost_matrix -= cost_matrix.min()

            # Apply Hungarian/Munkres algorithm to find optimal matches
            seg_labels, seg_matches = linear_sum_assignment(cost_matrix)

            # Assign results of matching to each segment's RegionProperties obj
            for i in range(count_prev):
                # Condition if segment has disappeared (assignment = "[D]")
                if seg_labels[i] == seg_matches[i]:
                    self.seg_properties[1][i].__match = "D"

                # Condition if segment has match (assignment = "[!]")
                if seg_labels[i] != seg_matches[i]:
                    j = seg_matches[i] - count_prev  # Offset (see matrix eg.)
                    self.seg_properties[1][i].__match = j
                    self.seg_properties[0][j].__match = i
            for i in range(count_curr):
                # Condition if segment has appeared (assignment = "[A]")
                if seg_labels[i+count_prev] == seg_matches[i+count_prev]:
                    self.seg_properties[0][i].__match = "A"
        else:
            seg_matches = []  # Empty list for visuals in case count_total = 0

        # Create visualization of segment matches if requested
        if visual:
            match_visualization()

    def analyse_matches(self):
        """Use matching results to store history of RegionProperties through
        chain of matches."""

        for seg_curr in self.seg_properties[0]:
            # Create an empty list to be filled with centroid history
            seg_curr.__centroids = []

            # This condition indicates a current-frame segment has appeared
            if seg_curr.__match == "A":
                # Append centroid value to end of list
                rounded_c = tuple([round(x, 3) for x in seg_curr.centroid])
                seg_curr.__centroids.append(rounded_c)

            # This condition indicates a current-frame segment has a match
            else:
                # Append past centroid values to list first
                seg_prev = self.seg_properties[1][seg_curr.__match]
                for c in seg_prev.__centroids:
                    seg_curr.__centroids.append(c)

                # Then append centroid value to end of list
                rounded_c = tuple([round(x, 3) for x in seg_curr.centroid])
                seg_curr.__centroids.append(rounded_c)

        for seg_prev in self.seg_properties[1]:
            # This condition indicates a previous-frame segment has disappeared
            if seg_prev.__match == "D":
                roi_value = (self.roi_mask[int(seg_prev.centroid[0])]
                                          [int(seg_prev.centroid[1])])

                # Valid "possible swift entering" event conditions:
                # 1: Centroid in ROI, 2: present for at least 2 frames
                if roi_value == 255 and len(seg_prev.__centroids) > 1:
                    # Storing information about event for further analysis
                    event_info = {
                        "TMSTAMP": self.timestamps[-1],
                        "FRM_NUM": self.framenumbers[-1],
                        "CENTRDS": seg_prev.__centroids,
                    }
                    self.event_list.append(event_info)


def extract_frames(args, queue_size=1, save_directory=None):
    """Function which uses object methods to extract individual frames
     (one at a time) from a video file. Saves each frame to image files for
     future reuse."""

    if not save_directory:
        save_directory = args.default_dir

    fq = FrameQueue(args, queue_size)

    print("[*] Reading frames... (This may take a while!)")
    while fq.frames_read < fq.src_framecount:
        success = fq.load_frame()

        if success:
            fq.save_frame(save_directory)
        else:
            raise Exception("read_frame() failed before expected end of file.")

        if fq.frames_read % 1000 == 0:
            print("[-] {}/{} frames successfully processed."
                  .format(fq.frames_read, fq.src_framecount))
            # NOTE: Should probably have some sort of "verbose" flag, or
            # utilize logging. This seems like a naive way to provide updates.
    fq.stream.release()
    print("[-] Extraction complete. {} total frames extracted."
          .format(fq.frames_read))


def process_frames(args, params):
    """Function which uses object methods to analyse a sequence of previously
    extracted frames and determine bird counts for that sequence."""

    def create_dataframe(passed_list):
        if passed_list:
            dataframe = pd.DataFrame(passed_list,
                                     columns=list(passed_list[0].keys())
                                     ).astype('object')
            dataframe["TMSTAMP"] = pd.to_datetime(dataframe["TMSTAMP"])
            dataframe["TMSTAMP"] = dataframe["TMSTAMP"].dt.round('us')
            dataframe.set_index(["TMSTAMP", "FRM_NUM"], inplace=True)
        else:
            dataframe = pd.DataFrame(passed_list)

        return dataframe

    print("[*] Analysing frames... (This may take a while!)")

    fq = FrameQueue(args, queue_size=params["queue_size"])
    while fq.frames_processed < fq.total_frames:
        if fq.frames_read < (params["queue_size"]-1):
            fq.load_frame(args.default_dir)
            fq.preprocess_frame()
        elif (params["queue_size"]-1) <= fq.frames_read < fq.total_frames:
            fq.load_frame(args.default_dir)
            fq.preprocess_frame()
            fq.segment_frame(args, params)
            # fq.match_segments(args, params)
            # fq.analyse_matches()
        elif fq.frames_read == fq.total_frames:
            fq.load_frame(empty=True)
            fq.segment_frame(args, params)
            # fq.match_segments(args, params)
            # fq.analyse_matches()

        if fq.frames_processed % 25 is 0 and fq.frames_processed is not 0:
            print("[-] {0}/{1} frames processed."
                  .format(fq.frames_processed, fq.total_frames))

    print("[-] Analysis complete. {0}/{1} frames were used in processing."
          .format(fq.frames_processed, fq.frames_read))

    df_eventinfo = create_dataframe(fq.event_list)

    return df_eventinfo


def full_algorithm(args, params, video_dict):
    def create_dataframe(passed_list):
        dataframe = pd.DataFrame(passed_list,
                                 columns=list(passed_list[0].keys())
                                 ).astype('object')
        dataframe["TMSTAMP"] = pd.to_datetime(dataframe["TMSTAMP"])
        dataframe["TMSTAMP"] = dataframe["TMSTAMP"].dt.round('us')
        dataframe.set_index(["TMSTAMP", "FRM_NUM"], inplace=True)

        return dataframe

    args.save_directory = args.default_dir
    args.chimney = video_dict["corners"]
    args.load = [0, 3000]

    fq = FrameQueue(args, params["queue_size"])

    while fq.frames_processed < fq.total_frames:
        if fq.frames_read < (params["queue_size"]-1):
            fq.load_frame()
            fq.preprocess_frame()
            # No segmentation needed until queue is filled
            # No matching needed until queue is filled
            # No analysis needed until queue is filled

        elif (params["queue_size"]-1) <= fq.frames_read < fq.total_frames:
            fq.load_frame()
            fq.preprocess_frame()
            fq.segment_frame(args, params)
            fq.match_segments(args, params)
            fq.analyse_matches()

        elif fq.frames_read == fq.total_frames:
            fq.load_frame(empty=True)
            # No preprocessing needed for empty frame
            fq.segment_frame(args, params)
            fq.match_segments(args, params)
            fq.analyse_matches()

        if fq.frames_processed % 25 is 0 and fq.frames_processed is not 0:
            print("[-] {0}/{1} frames processed."
                  .format(fq.frames_processed, fq.total_frames))

    df_eventinfo = create_dataframe(fq.event_list)

    return df_eventinfo
