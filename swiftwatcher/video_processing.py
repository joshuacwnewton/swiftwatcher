# Bugfix for PyInstaller, see: https://github.com/numpy/numpy/issues/14163
import numpy.random.common
import numpy.random.bounded_integers
import numpy.random.entropy

# Imports used in numerous stages
import numpy as np
import cv2
import collections
import math
from os import fspath

# Necessary imports for segmentation stage
from scipy import ndimage as img
from utils.rpca_ialm import inexact_augmented_lagrange_multiplier

# Necessary imports for matching stage
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from skimage import measure
import pandas as pd
import sys
eps = sys.float_info.epsilon


class FrameQueue:
    """Class which loads frames into a temporary queue, while also providing
    methods for processing these frames. The methods can be used to detect
    events where chimney swifts may have entered an in-frame chimney."""

    def __init__(self, config, queue_size=21):
        def assign_file_properties():
            """Assign properties related to the source video file, including
            file properties and video frame properties."""
            
            self.src_filepath = config["src_filepath"]
            if not self.src_filepath.exists():
                raise Exception(
                    "[!] Filepath does not point to valid video file.")

            # Open source video file and initialize its immutable attributes
            self.stream = cv2.VideoCapture(fspath(self.src_filepath))
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
                self.src_starttime = pd.Timestamp(config["timestamp"])

            # Separate from src for eventual cropping/resizing
            self.height = self.src_height
            self.width = self.src_width

        def assign_processing_properties():
            """Assign properties relating to frame processing, including
            motion analysis, segmentation, and event detection stages."""
            
            # Initialize primary queue for unaltered frames
            self.queue_size = queue_size
            self.queue = collections.deque([], queue_size)
            self.framenumbers = collections.deque([], queue_size)
            self.timestamps = collections.deque([], queue_size)
            self.frames_read = 0
            self.frames_processed = 0

            # Initialize secondary queues for segmented frames (size=2 because
            # when tracking swifts, only 2 frames are compared at a time.)
            self.seg_queue = collections.deque([], 2)
            self.seg_properties = collections.deque([], 2)
            # Append empty values because otherwise the frame comparison would
            # fail when only the first frame has been loaded.
            self.seg_queue.appendleft(np.zeros((self.height, self.width))
                                      .astype(np.uint8))
            self.seg_properties.appendleft([])

            # Store detected events in this list
            self.event_list = []

            # Keep track of sequential errors to break if necessary
            self.failcount = 0

        def generate_chimney_regions():
            """Generate rectangular regions from two corners of chimney:

                         (x1, y1) *-----------------* (x2, y2)
                                  |                 |
                                  |  chimney stack  |
                                  |                 |

            Regions will be used for cropping and generating chimney ROI."""

            # From provided corners, determine which coordinates are the
            # outer-most ones.
            left = min(config["corners"][0][0], config["corners"][1][0])
            right = max(config["corners"][0][0], config["corners"][1][0])
            bottom = max(config["corners"][0][1], config["corners"][1][1])

            width = right - left

            # Dimensions = (1.25 X 0.625) = (2 X 1) ratio of width to height
            self.crop_region = [(left - int(0.125*width),
                                 bottom - int(0.5 * width)),
                                (right + int(0.125 * width),
                                 bottom + int(0.125 * width))]

            # Left and right brought in slightly as swifts don't enter at edge
            self.roi_region = [(int(left + 0.025 * width),
                                int(bottom - 0.25 * width)),
                               (int(right - 0.025 * width),
                                int(bottom))]

        def generate_roi_mask():
            """Generate a mask that contains the chimney's region of interest.
            The "roi_region" is determined in generate_chimney_regions()."""

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

            # Add roi to empty image of the same size as the frame
            frame_with_thr = np.zeros_like(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            frame_with_thr[self.roi_region[0][1]:self.roi_region[1][1],
                           self.roi_region[0][0]:self.roi_region[1][0]] = thr

            # Apply preprocessing to ROI mask image (identical to what would
            # be applied to the frames themselves)
            frame_with_thr = self.preprocess_frame(frame_with_thr)
            _, self.roi_mask = cv2.threshold(frame_with_thr, 0, 255,
                                             cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        assign_file_properties()
        assign_processing_properties()
        generate_chimney_regions()
        generate_roi_mask()

    def load_frame(self, blank=False):
        """Load new frame into left side (index 0) of queue."""

        def fn_to_ts(frame_number):
            """Helper function to convert frame amount into a timestamp."""
            total_s = frame_number / self.src_fps
            timestamp = self.src_starttime + pd.Timedelta(total_s, 's')

            return timestamp

        # Used when queue has to be advanced but there are no more frames left.
        if blank:
            self.timestamps.appendleft("")
            self.framenumbers.appendleft("")
            self.queue.appendleft(np.array([]))
            success = True

        # By default, read frames from video file.
        else:
            new_framenumber = int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))
            new_timestamp = fn_to_ts(self.stream.get(cv2.CAP_PROP_POS_FRAMES))
            success, frame = self.stream.read()
            if success:
                self.frames_read += 1
                self.timestamps.appendleft(new_timestamp)
                self.framenumbers.appendleft(new_framenumber)
                self.queue.appendleft(frame)

        return success

    def preprocess_frame(self, frame=None, index=0):
        """Apply preprocessing to frame prior to motion analysis."""

        def convert_grayscale():
            """Convert a frame from 3-channel RGB to grayscale."""
            nonlocal frame

            if len(frame.shape) is 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        def crop_frame():
            """Crop frame to dimensions specified in __init__ of FrameQueue."""
            nonlocal frame

            corners = self.crop_region

            frame = frame[corners[0][1]:corners[1][1],
                          corners[0][0]:corners[1][0]]

            # Update frame attributes
            self.height = frame.shape[0]
            self.width = frame.shape[1]

        def resize_frame():
            """Resize frame so dimensions are fixed regardless of chimney
            size. (Different chimneys produce different crop dimensions.)"""
            nonlocal frame

            frame = cv2.resize(frame, (300, 150))

            # Update frame attributes
            self.height = frame.shape[0]
            self.width = frame.shape[1]

        # Used when creating ROI mask (see __init__)
        if frame is not None:
            convert_grayscale()
            crop_frame()
            resize_frame()
            return frame

        # Used when processing the video's frames to determine swift counts
        else:
            frame = self.queue[index]
            convert_grayscale()
            crop_frame()
            resize_frame()
            self.queue[index] = frame

    def segment_frame(self):
        """Take the last frame of the queue and process it so birds are
        segmented from the background. Store segmented frame (and its
        properties) in a secondary queue."""

        def rpca(index=None):
            """Decompose set of images into corresponding low-rank and sparse
            images. Method expects images to have been reshaped to matrix of
            column vectors.

            Note: frame = lowrank + sparse, where:
                          lowrank = "background" image
                          sparse  = "foreground" errors corrupting
                                    the "background" image

            The size of the queue will determine the tradeoff between 
            computational efficiency and accuracy."""

            # Reshape frames into column vector matrix, 1 vector for each frame
            img_matrix = np.array(self.queue)
            if index:
                img_matrix = img_matrix[:index, :, :]
            col_matrix = np.transpose(img_matrix.reshape(img_matrix.shape[0],
                                                         img_matrix.shape[1] *
                                                         img_matrix.shape[2]))

            # Algorithm for the IALM approximation of Robust PCA method.
            lr_columns, s_columns = \
                inexact_augmented_lagrange_multiplier(col_matrix)

            # Bring pixels that are darker than background to [0, 255] range
            s_columns = np.negative(s_columns)
            s_columns = np.clip(s_columns, 0, 255).astype(np.uint8)

            # Reshape columns back into image dimensions and store in queue
            for i in range(img_matrix.shape[0]):
                self.queue[i] = np.reshape(s_columns[:, i],
                                           (self.height, self.width))

        def filter_rpca_output():
            """Take raw RPCA output (demonstrating motion in frame) and filter
            out any non-swift motion."""

            rpca_output = self.queue[-1]
            smoothed_frame = cv2.bilateralFilter(rpca_output,
                                                 d=7,
                                                 sigmaColor=15,
                                                 sigmaSpace=1)
            _, thresholded_frame = cv2.threshold(smoothed_frame,
                                                 thresh=15, maxval=255,
                                                 type=cv2.THRESH_TOZERO)
            opened_frame = img.grey_opening(thresholded_frame,
                                            size=(3, 3)).astype(np.uint8)

            return opened_frame

        def store_segmentation(frame):
            """Apply connected component labeling, and store the image (and
            its properties) in the secondary segmentation queues."""
            # Segment using CC labeling
            _, labeled_frame = cv2.connectedComponents(frame, connectivity=4)

            # Append segmented frame (and information about frame) to queue
            self.seg_queue.appendleft(labeled_frame.astype(np.uint8))
            self.seg_properties.appendleft(measure.regionprops(labeled_frame))
            self.frames_processed += 1

        # Partial batch RPCA for remainder frames (total frames % queue size)
        if self.frames_read == self.src_framecount:
            if self.frames_read - self.frames_processed == self.queue_size:
                rem = self.src_framecount % self.queue_size
                rpca(index=rem)

        # Full batch RPCA (elif to prevent this from double-triggering if
        # total video length happens to be a multiple of the queue size)
        elif self.frames_read % self.queue_size == 0:
            rpca()

        filtered = filter_rpca_output()
        store_segmentation(filtered)

    def match_segments(self):
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

                          Cost Matrix

        -[!]: Costs associated with two segments matching. (A more unlikely
            match has a higher associated cost.)
        -[D]/[A]: Costs associated with two segments having no match.
            (Disappeared and appeared respectively.)
        -[ ]: Impossible spaces. (These are filled with values such that it is
            impossible for them to be chosen by the matching algorithm.

        By using the Hungarian algorithm, this matrix is solved to give an
        ideal outcome (match, appear, disappear) for each segment in both
        frames. The total sum of selected costs will be the minimum across all
        possible combinations of matches, as per the Assignment Problem.

        An example match could be: 0 (prev) -> 0 (curr) (! value selected),
                                   1 (prev) -> 2 (curr) (! value selected),
                                   2 (prev) -> 3 (curr) (! value selected),
                                   and 1 (curr) appears (A value selected)."""

        def generate_cost_matrix():
            """Compute entries in the cost matrix, corresponding to the
            diagram provided in the match_segments() docstring."""

            # Initialize cost matrix as (N+M)x(N+M)
            cost_matrix = (1+eps)*np.ones((count_total, count_total))

            # Matrix values: likelihood of segments being a match
            for seg_prev in self.seg_properties[1]:
                for seg in self.seg_properties[0]:
                    # Convert segment labels to cost matrix indices
                    index_v = (seg_prev.label - 1)
                    index_h = (count_prev + seg.label - 1)

                    # Compute "distance cost".
                    # (distances > 20px will have much higher costs)
                    dist = distance.euclidean(seg_prev.centroid,
                                              seg.centroid)
                    dist_cost = 2 ** (dist - 25)

                    # Compute "angle cost" if previous angle exists.
                    # (angles > 90* will have much higher costs.)
                    if len(seg_prev.__centroids) > 1:
                        centroid_list = seg_prev.__centroids

                        del_y_full = centroid_list[0][0] - centroid_list[-1][0]
                        del_x_full = -1 * (
                                    centroid_list[0][1] - centroid_list[-1][1])
                        angle_prev = math.degrees(math.atan2(del_y_full,
                                                             del_x_full))

                        del_y_new = centroid_list[-1][0] - seg.centroid[0]
                        del_x_new = -1 * (
                                    centroid_list[-1][1] - seg.centroid[1])
                        angle = math.degrees(math.atan2(del_y_new, del_x_new))

                        angle_diff = min(360 - abs(angle - angle_prev),
                                         abs(angle-angle_prev))

                        angle_cost = 2**(angle_diff - 90)
                    else:
                        angle_cost = 1

                    # Average both costs to get cost matrix entry
                    cost = 0.5*(dist_cost+angle_cost)
                    cost_matrix[index_v, index_h] = cost

            # Matrix values: likelihood of segments having no match
            for i in range(count_total):
                cost_matrix[i, i] = 1

            return cost_matrix

        def assign_labels(cost_matrix):
            """ Assign results of Hungarian algorithm's matching to the
            RegionProperties object for each segment"""

            seg_labels, seg_matches = linear_sum_assignment(cost_matrix)

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

        count_curr = len(self.seg_properties[0])
        count_prev = len(self.seg_properties[1])
        count_total = count_curr + count_prev

        if count_total > 0:
            costs = generate_cost_matrix()
            assign_labels(costs)

    def analyse_matches(self):
        """Analyse matching results to do two things:
            -Store centroid history in segment's RegionProperties object
            -Determine whether a disappearing segment meets "event detection"
            criteria.."""

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


def select_corners(filepath):
    """OpenCV GUI function to select chimney corners from video frame."""

    def click_and_update(event, x, y, flags, param):
        """Callback function to record mouse coordinates on click, and to
        update instructions to user."""
        nonlocal corners

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(corners) < 2:
                corners.append((int(x), int(y)))
                cv2.circle(image, corners[-1], 5, (0, 0, 255), -1)
                cv2.imshow("image", image)
                cv2.resizeWindow('image',
                                 int(0.5*image.shape[1]),
                                 int(0.5*image.shape[0]))

            if len(corners) == 1:
                cv2.setWindowTitle("image",
                                   "Click on corner 2")

            if len(corners) == 2:
                cv2.setWindowTitle("image",
                                   "Type 'y' to keep,"
                                   " or 'n' to select different corners.")

    stream = cv2.VideoCapture(fspath(filepath))
    success, image = stream.read()
    clone = image.copy()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", click_and_update)
    cv2.setWindowTitle("image", "Click on corner 1")

    corners = []

    while True:
        # Display image and wait for user input (click -> click_and_update())
        cv2.imshow("image", image)
        cv2.resizeWindow('image',
                         int(0.5 * image.shape[1]),
                         int(0.5 * image.shape[0]))
        cv2.waitKey(1)

        # Condition for when two corners have been selected
        if len(corners) == 2:
            key = cv2.waitKey(2000) & 0xFF

            if key == ord("n") or key == ord("N"):
                # Indicates selected corners are not good, so resets state
                image = clone.copy()
                corners = []
                cv2.setWindowTitle("image",
                                   "Click on corner 1")

            elif key == ord("y") or key == ord("Y"):
                # Indicates selected corners are acceptable
                break

        if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) == 0:
            # Indicates window has been closed
            corners = []
            break

    cv2.destroyAllWindows()

    return corners


def swift_counting_algorithm(config):
    """Full algorithm which uses FrameQueue methods to process an entire video
    from start to finish."""

    def create_dataframe(passed_list):
        """Convert list of events to pandas dataframe."""
        dataframe = pd.DataFrame(passed_list,
                                 columns=list(passed_list[0].keys())
                                 ).astype('object')
        dataframe["TMSTAMP"] = pd.to_datetime(dataframe["TMSTAMP"])
        dataframe["TMSTAMP"] = dataframe["TMSTAMP"].dt.round('us')
        dataframe.set_index(["TMSTAMP", "FRM_NUM"], inplace=True)

        return dataframe

    print("[*] Now processing {}.".format(config["name"]))
    # print("[-]     Status updates will be given every 100 frames.")

    fq = FrameQueue(config)
    while fq.frames_processed < fq.src_framecount:
        success = False

        # Store state variables in case video processing glitch occurs
        # (e.g. due to poorly encoded video)
        pos = fq.stream.get(cv2.CAP_PROP_POS_FRAMES)
        read = fq.frames_read
        proc = fq.frames_processed

        try:
            # Load frames until queue is filled
            if fq.frames_read < (fq.queue_size - 1):
                success = fq.load_frame()
                fq.preprocess_frame()
                # fq.segment_frame() (not needed until queue is filled)
                # fq.match_segments() (not needed until queue is filled)
                # fq.analyse_matches() (not needed until queue is filled)

            # Process queue full of frames
            elif (fq.queue_size - 1) <= fq.frames_read < fq.src_framecount:
                success = fq.load_frame()
                fq.preprocess_frame()
                fq.segment_frame()
                fq.match_segments()
                fq.analyse_matches()

            # Load blank frames until queue is empty
            elif fq.frames_read == fq.src_framecount:
                success = fq.load_frame(blank=True)
                # fq.preprocess_frame() (not needed for blank frame)
                fq.segment_frame()
                fq.match_segments()
                fq.analyse_matches()

        except Exception as e:
            # TODO: Print statements here should be replaced with logging
            # Previous: print("[!] Error has occurred, see: '{}'.".format(e))
            # I'm not satisfied with how unexpected errors are handled
            fq.failcount += 1

            # Increment state variables to ensure algorithm doesn't get stuck
            if fq.stream.get(cv2.CAP_PROP_POS_FRAMES) == pos:
                fq.stream.grab()
            if fq.frames_read == read:
                fq.frames_read += 1
            if fq.frames_processed == proc:
                fq.frames_processed += 1

        if success:
            fq.failcount = 0
        else:
            fq.failcount += 1

        # Break if too many sequential errors
        if fq.failcount >= 10:
            # TODO: Print statements here should be replaced with logging
            # Previous: print("[!] Too many sequential errors have occurred. "
            #                 "Halting algorithm...")
            # I'm not satisfied with how unexpected errors are handled
            fq.frames_processed = fq.src_framecount + 1

        # Status updates
        if fq.frames_processed % 25 is 0 and fq.frames_processed is not 0:
            sys.stdout.write("\r[-]     {0}/{1} frames processed.".format(
                fq.frames_processed, fq.src_framecount))
            sys.stdout.flush()

    if fq.event_list:
        df_eventinfo = create_dataframe(fq.event_list)
    else:
        df_eventinfo = []
    print("")

    return df_eventinfo
