# Stdlib imports
import os
import glob
import collections
from time import sleep

# Imports used in numerous stages
import cv2
import numpy as np
import utils.cm as cm  # Used for colormapping, not entirely necessary

# Necessary imports for segmentation stage
from scipy import ndimage as img
from utils.rpca_ialm import inexact_augmented_lagrange_multiplier

# Necessary imports for matching stage
import math
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from skimage import measure


class FrameQueue:
    """Class for storing, describing, and manipulating frames from video file
    using FIFO queues. Essentially deques with additional attributes and
    methods specific to video analysis.

    Example FrameQueue object of size 7:

    framenumbers: [571][570][569][568][567][566][565]
    queue:        [i_0][i_1][i_2][i_3][i_4][i_5][i_6] (primary)
    seg_queue:                   [i_0][i_1][i_2][i_3] (secondary)

    The primary queue ("queue") stores original frames, and the secondary queue
    ("seg_queue") stores segmented versions of the frames. As segmentation
    requires contextual information from past/future frames, the center index
    of the primary queue (index 3) will correspond to the 0th index in the
    secondary queue."""
    def __init__(self, video_directory, filename, queue_size=1,
                 desired_fps=False):
        # Check validity of filepath
        video_filepath = video_directory + filename
        if not os.path.isfile(video_filepath):
            raise Exception("[!] Filepath does not point to valid video file.")

        # Open source video file and store its immutable attributes
        self.src_filename = filename
        self.src_directory = video_directory
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

        # Initialize mutable frame/video attributes
        self.height = self.src_height  # Separate b/c dimensions may change
        self.width = self.src_width    # Separate b/c dimensions may change
        if not desired_fps:
            self.fps = self.src_fps
            self.delay = round(self.src_fps / self.fps) - 1
        else:
            self.fps = desired_fps
            self.delay = 0  # Delay parameter needed to subsample video

        # Initialize primary queue for unaltered frames
        self.queue = collections.deque([], queue_size)
        self.framenumbers = collections.deque([], queue_size)
        self.timestamps = collections.deque([], queue_size)
        self.frames_read = 0

        # Initialize secondary queue for segmented frames
        self.queue_center = int((queue_size - 1) / 2)
        self.seg_queue = collections.deque([], self.queue_center)
        self.seg_properties = collections.deque([], self.queue_center)

    def load_frame_from_video(self):
        """Insert next frame from stream into left side (index 0) of queue."""
        if not self.stream.isOpened():
            raise Exception("[!] Video stream is not open."
                            " Cannot read new frames.")

        # Fetch new frame and update its attributes
        self.framenumbers.appendleft(
            int(self.stream.get(cv2.CAP_PROP_POS_FRAMES)))
        self.timestamps.appendleft(
            (ms_to_timestamp(self.stream.get(cv2.CAP_PROP_POS_MSEC))))
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
        timestamp = framenumber_to_timestamp(frame_number, self.fps)
        self.timestamps.appendleft(timestamp)

        # Set appropriate directory
        if not folder_name:
            time = self.timestamps[0].split(":")
            save_directory = base_save_directory+"/frames/"+time[0]+":"+time[1]
        else:
            save_directory = base_save_directory+"/"+folder_name

        # Load frame from file
        file_paths = glob.glob("{0}/frame{1}_{2}*".format(save_directory,
                                                          frame_number,
                                                          timestamp))
        frame = cv2.imread(file_paths[0])
        self.queue.appendleft(np.array(frame))

        if not frame.size == 0:
            success = True
            self.frames_read += 1
        else:
            success = False

        return success

    def save_frame_to_file(self, base_save_directory, frame=None, index=0,
                           folder_name=None, file_prefix="", file_suffix="",
                           scale=100):
        """Save an individual frame to an image file. If frame itself is not
        provided, frame will be pulled from frame_queue at specified index."""

        # By default, frames will be saved in a subfolder corresponding to
        # HH:MM formatting. However, a custom subfolder can be chosen instead.
        if not folder_name:
            time = self.timestamps[index].split(":")
            save_directory = base_save_directory+"/frames/"+time[0]+":"+time[1]
        else:
            save_directory = base_save_directory+"/"+folder_name

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
        if scale is not 100:
            s = scale / 100
            frame = cv2.resize(frame,
                               (round(frame.shape[1]*s),
                                round(frame.shape[0]*s)),
                               interpolation=cv2.INTER_AREA)

        # Write frame to image file within save_directory
        try:
            cv2.imwrite("{0}/{1}frame{2}_{3}{4}.jpg"
                        .format(save_directory, file_prefix,
                                self.framenumbers[index],
                                self.timestamps[index],
                                file_suffix),
                        frame)
        except Exception as e:
            print("[!] Image saving failed due to: {0}".format(str(e)))

    def convert_grayscale(self, index=0, algorithm="cv2 built-in"):
        """Convert to grayscale a frame at specified index of FrameQueue"""
        if algorithm == "cv2 default":
            self.queue[index] = cv2.cvtColor(self.queue[index],
                                             cv2.COLOR_BGR2GRAY)

    def crop_frame(self, corners, index=0):
        """Crop frame at specified index of FrameQueue."""
        try:
            self.queue[index] = self.queue[index][corners[0][1]:corners[1][1],
                                                  corners[0][0]:corners[1][0]]
        except Exception as e:
            print("[!] Frame cropping failed due to: {0}".format(str(e)))

        # Update frame attributes if necessary
        height = corners[1][1] - corners[0][1]
        width = corners[1][0] - corners[0][0]
        if height is not self.height:
            self.height = height
        if width is not self.width:
            self.width = width

    def frame_to_column(self, index=0):
        """Reshapes an NxM frame into an (N*M)x1 column vector."""
        self.queue[index] = np.squeeze(np.reshape(self.queue[index],
                                                  (self.width*self.height, 1)))

    def rpca(self, lmbda, tol, maxiter, darker, index=0):
        """Decompose set of images into corresponding low-rank and sparse
        images. Method expects images to have been reshaped to matrix of
        column vectors.

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

    def segment_frame(self, lmbda, tol, maxiter, darker,
                      iters, diameter, sigma_color, sigma_space,
                      thr_value, thr_type,
                      gry_op_SE, segmentation,
                      index, visual=False):
        """Segment birds from one frame ("index") using information from other
        frames in the FrameQueue object. Store segmented frame in secondary
        queue."""
        # Apply Robust PCA method to isolate regions of motion
        # lowrank = "background" image
        # sparse  = "foreground" errors which corrupts the "background" image
        # frame = lowrank + sparse
        lowrank, sparse = self.rpca(lmbda, tol, maxiter, darker, index)

        # Apply bilateral filter to smooth over low-contrast regions
        sparse_filtered = sparse
        for i in range(iters):
            sparse_filtered = cv2.bilateralFilter(sparse_filtered, diameter,
                                                  sigma_color, sigma_space)

        # Apply thresholding to retain strongest areas and discard the rest
        _, sparse_thr = cv2.threshold(sparse_filtered,
                                      thresh=thr_value,
                                      maxval=255,
                                      type=thr_type)

        # Discard areas where 2x2 structuring element will not fit
        sparse_opened = \
            img.grey_opening(sparse_thr, size=gry_op_SE) \
            .astype(sparse_thr.dtype)

        # Segment using connected component labeling
        num_components, sparse_cc = eval(segmentation)
        # num_components, sparse_cc = \
        #     cv2.connectedComponents(sparse_opened, connectivity=conn)

        # Append empty values first if queue is empty
        if self.seg_queue.__len__() is 0:
            self.seg_queue.appendleft(np.zeros((self.height, self.width))
                                      .astype(np.uint8))
            self.seg_properties.appendleft([])

        # Append segmented frame (and information about frame) to queue
        self.seg_queue.appendleft(sparse_cc.astype(np.uint8))
        self.seg_properties.appendleft(measure.regionprops(sparse_cc))

        # Create visualization of processing stages if requested
        if visual:
            # Fetch unprocessed frame, reshape into image from column vector
            frame = np.reshape(self.queue[index], (self.height, self.width))

            # Scale labeled image to be visible with uint8 grayscale
            if num_components > 0:
                sparse_cc_scaled = sparse_cc*int(255/num_components)

            # Combine each stage into one image, separated for visual clarity
            separator = 64*np.ones(shape=(1, self.width), dtype=np.uint8)
            processing_stages = np.vstack((frame, separator,
                                           sparse, separator,
                                           sparse_filtered, separator,
                                           sparse_thr, separator,
                                           sparse_opened, separator,
                                           sparse_cc_scaled)).astype(np.uint8)
        else:
            processing_stages = None
        return processing_stages

    def match_segments(self, match_function, notmatch_function,
                       index=0, visual=False):
        """Analyze a pair of segmented frames and return conclusions about
        which segments match between frames."""
        # Assign name to commonly used properties
        count = len(self.seg_properties[index])
        count_prev = len(self.seg_properties[index+1])
        count_total = count + count_prev

        # Initialize coordinates and stats as empty
        coords = [[(0, 0) for col in range(2)] for row in range(count_total)]
        stats = {
            "total_matches": 0,
            "appeared_from_edge": 0,
            "appeared_from_chimney": 0,
            "disappeared_to_edge": 0,
            "disappeared_to_chimney": 0,
            "anomalies": 0
        }

        # Compute and analyze match pairs only if bird segments exist
        if count_total > 0:
            # Initialize likelihood matrix
            likeilihood_matrix = np.zeros((count_total, count_total))

            # Matrix values: likelihood of segments being a match
            for seg_prev in self.seg_properties[index+1]:
                for seg in self.seg_properties[index]:
                    # Convert segment labels to likelihood matrix indices
                    index_v = (seg_prev.label - 1)
                    index_h = (count_prev + seg.label - 1)

                    # Likeilihoods as a function of distance between segments
                    dist = distance.euclidean(seg_prev.centroid,
                                              seg.centroid)
                    # Map distance values using a Gaussian curve
                    likeilihood_matrix[index_v, index_h] = eval(match_function)
                    # likeilihood_matrix[index_v, index_h] = \
                    #     math.exp(-1 * (((dist - 10) ** 2) / 40))

            # Matrix values: likelihood of segments appearing/disappearing
            for i in range(count_total):
                # Compute closest distance from segment to edge of frame
                if i < count_prev:
                    point = self.seg_properties[index+1][i].centroid
                if count_prev <= i < (count + count_prev):
                    point = self.seg_properties[index][i - count_prev].centroid
                edge_distance = min([point[0], point[1],
                                     self.height - point[0],
                                     self.width - point[1]])

                # Map distance values using an Exponential curve
                likeilihood_matrix[i, i] = eval(notmatch_function)
                # likeilihood_matrix[i, i] = (1 / 2) * math.exp(
                #     -edge_distance / 10)

            # Convert likelihood matrix into cost matrix
            cost_matrix = -1*likeilihood_matrix
            cost_matrix -= cost_matrix.min()

            # Apply Hungarian/Munkres algorithm to find optimal matches
            _, matches = linear_sum_assignment(cost_matrix)

            # Convert matches (pairs of indices) into pairs of coordinates
            for i in range(count_total):
                j = matches[i]
                # Note: cv2.line requires int, .centroid returns float
                # Conversion must happen, hence intermediate "float_coord"

                # Index condition if two segments are matching
                if (i < j) and (i < count_prev):
                    float_coord1 = \
                        self.seg_properties[index+1][i].centroid
                    float_coord2 = \
                        self.seg_properties[index][j - count_prev].centroid
                    coords[i][0] = tuple([int(val) for val in float_coord1])
                    coords[i][1] = tuple([int(val) for val in float_coord2])
                # Index condition for when a previous segment has disappeared
                if (i == j) and (i < count_prev):
                    float_coord1 = \
                        self.seg_properties[index+1][i].centroid
                    coords[i][0] = tuple([int(val) for val in float_coord1])
                # Index condition for when a new segment has appeared
                if (i == j) and (i >= count_prev):
                    float_coord2 = \
                        self.seg_properties[index][j - count_prev].centroid
                    coords[i][1] = tuple([int(val) for val in float_coord2])

            # For each pair of coordinates, classify as certain behaviors
            for coord_pair in coords:
                # If this condition is met, pair means a segment appeared
                if coord_pair[0] == (0, 0) and coord_pair[1] == (0, 0):
                    pass
                elif coord_pair[0] == (0, 0):
                    edge_distance = min(coord_pair[1][0], coord_pair[1][1],
                                        self.width - coord_pair[1][1])
                    chimney_distance = self.height - coord_pair[1][0]

                    if edge_distance <= 10:
                        stats["appeared_from_edge"] += 1
                    elif chimney_distance <= 10:
                        stats["appeared_from_chimney"] += 1
                    else:
                        stats["anomalies"] += 1
                # If this condition is met, pair means a segment disappeared
                elif coord_pair[1] == (0, 0):
                    edge_distance = min(coord_pair[0][0], coord_pair[0][1],
                                        self.width - coord_pair[0][1])
                    chimney_distance = self.height - coord_pair[0][0]

                    if edge_distance <= 10:
                        stats["disappeared_to_edge"] += 1
                    elif chimney_distance <= 10:
                        stats["disappeared_to_chimney"] += 1
                    else:
                        stats["anomalies"] += 1
                # Otherwise, a match was made
                else:
                    stats["total_matches"] += 1
                test = None

        # Create visualization of segment matches if requested
        if visual:
            # Scale labeled images to be visible with uint8 grayscale
            if count > 0:
                frame = self.seg_queue[index]*int(255/count)
            else:
                frame = self.seg_queue[index]

            if count_prev > 0:
                frame_prev = self.seg_queue[index+1]*int(255/count_prev)
            else:
                frame_prev = self.seg_queue[index + 1]

            # Combine both frames into single image
            separator_h = 64 * np.ones(shape=(self.height, 1),
                                       dtype=np.uint8)
            match_comparison = np.hstack((frame_prev, separator_h, frame))

            for coord_pair in coords:
                if np.count_nonzero(coord_pair) == 4:
                    # Note: cv2's point formatting is reversed to skimage's
                    cv2.line(match_comparison,
                             (coord_pair[0][1], coord_pair[0][0]),
                             (coord_pair[1][1] + self.width, coord_pair[1][0]),
                             color=(255, 255, 255), thickness=1)
        else:
            match_comparison = None

        return coords, stats, match_comparison


def ms_to_timestamp(total_ms):
    """Helper function to convert millisecond value into timestamp."""
    # cv2's VideoCapture class provides the position of the video in
    # milliseconds (cv2.CAP_PROP_POS_MSEC). However, as it can be easier to
    # think of video in terms of hours, minutes, and seconds, it's helpful to
    # be able to convert back and forth.
    total_s = int(total_ms/1000)
    total_m = int(total_s/60)
    total_h = int(total_m/60)

    ms = round(total_ms % 1000)
    s = round(total_s % 60)
    m = round(total_m % 60)
    h = round(total_h % 24)

    timestamp = "{0:02d}:{1:02d}:{2:02d}:{3:03d}".format(h, m, s, ms)
    return timestamp


def framenumber_to_timestamp(frame_number, fps):
    """Helper function to convert an amount of frames into a timestamp."""
    # cv2's VideoCapture class provides a frame count property
    # (cv2.CAP_PROP_FRAME_COUNT) but not a duration property. However, as it
    # can be easier to think of video in terms of hours, minutes, and seconds,
    # it's helpful to be able to convert back and forth.
    milliseconds = (frame_number / fps)*1000
    timestamp = ms_to_timestamp(milliseconds)

    return timestamp


def timestamp_to_framenumber(timestamp, fps):
    """Helper function to convert timestamp into an amount of frames."""
    # cv2's VideoCapture class provides a frame count property
    # (cv2.CAP_PROP_FRAME_COUNT) but not a duration property. However, as it
    # can be easier to think of video in terms of hours, minutes, and seconds,
    # it's helpful to be able to convert back and forth.
    time = timestamp.split(":")
    seconds = (float(time[0])*60*60 +
               float(time[1])*60 +
               float(time[2]) +
               float(time[3])/1000)
    frame_number = int(round(seconds * fps))
    return frame_number


def extract_frames(video_directory, filename, queue_size=1,
                   save_directory=None):
    """Helper function to extract individual frames one at a time 
       from video file and save them to image files for future reuse."""

    # Default save directory chosen to be identical to filename
    if not save_directory:
        save_directory = video_directory + os.path.splitext(filename)[0]

    print("[========================================================]")
    print("[*] Reading frames... (This may take a while!)")

    frame_queue = FrameQueue(video_directory, filename, queue_size)
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
