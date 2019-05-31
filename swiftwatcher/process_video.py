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
    using a FIFO queue. Essentially a deque with additional attributes and
    methods specific to video analysis.

    Queue is size 1 by default. Increase queue size for analysis requiring
    multiple sequential frames.

    Attributes (frame queue):  queue, framenumbers, timestamps, frames_read
    Attributes (source video): src_filename, src_directory, src_fps,
                               src_framecount, src_height, src_width, src_codec
    Attributes (frames): height, width, delay, fps

    Methods (File I/O): load_frame_from_video, load_frame_from_file,
                        save_frame_to_file,
    Methods (Frame processing): convert_grayscale, segment_frame, crop_frame,
                                resize_frame, frame_to_column,
                                rpca, segment_frame, match_segments"""

    def __init__(self, video_directory, filename, queue_size=1,
                 desired_fps=False):
        # Initialize queue attributes
        self.queue = collections.deque([], queue_size)
        self.framenumbers = collections.deque([], queue_size)
        self.timestamps = collections.deque([], queue_size)
        self.frames_read = 0

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
            save_directory = base_save_directory+"/"+time[0]+":"+time[1]
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
            save_directory = base_save_directory+"/"+time[0]+":"+time[1]
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

    def convert_grayscale(self, index):
        """Convert to grayscale a frame at specified index of FrameQueue"""
        try:
            self.queue[index] = cv2.cvtColor(self.queue[index],
                                             cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print("[!] Frame conversion failed due to: {0}".format(str(e)))

    def crop_frame(self, corners, index):
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

    def resize_frame(self, scale_percent, index=0):
        s = scale_percent/100
        self.queue[index] = cv2.resize(self.queue[index],
                                       (round(self.queue[index].shape[1]*s),
                                        round(self.queue[index].shape[0]*s)),
                                       interpolation=cv2.INTER_AREA)

    def frame_to_column(self, index):
        """Reshapes an NxM frame into an (N*M)x1 column vector."""
        self.queue[index] = np.squeeze(np.reshape(self.queue[index],
                                                  (self.width*self.height, 1)))

    def rpca(self, index, darker_only=False):
        """Decompose set of images into corresponding low-rank and sparse
        images. Method expects images to have been reshaped to matrix of
        column vectors.

        The size of the queue will determine the tradeoff between efficiency
        and accuracy."""

        # np.array alone would give an e.g. 20x12500x1 matrix. Adding transpose
        # and squeeze yields 12500x20. (i.e. a matrix of column vectors)
        matrix = np.squeeze(np.transpose(np.array(self.queue)))

        # Algorithm for the IALM approximation of Robust PCA method.
        lr_columns, s_columns = \
            inexact_augmented_lagrange_multiplier(matrix, verbose=False)

        # Slice frame from low rank and sparse and reshape back into image
        lr_image = np.reshape(lr_columns[:, index], (self.height, self.width))
        s_image = np.reshape(s_columns[:, index], (self.height, self.width))

        # Bring pixels that are darker than the background into [0, 255] range
        if darker_only:
            s_image = -1 * s_image  # Darker = negative -> mirror into positive
            np.clip(s_image, 0, 255, s_image)

        return lr_image.astype(dtype=np.uint8), s_image.astype(dtype=np.uint8)

    def segment_frame(self, index, visual=False):
        """Segment birds from one frame ("index") using information from other
        frames in the FrameQueue object."""
        # Apply Robust PCA method to isolate regions of motion
        # lowrank = "background" image
        # sparse  = "foreground" errors which corrupts the "background" image
        # frame = lowrank + sparse
        lowrank, sparse = self.rpca(index, darker_only=True)

        # Apply bilateral filter to smooth over low-contrast regions
        sparse_filtered = sparse
        for i in range(2):
            sparse_filtered = cv2.bilateralFilter(sparse_filtered,
                                                  d=7,
                                                  sigmaColor=15,
                                                  sigmaSpace=1)

        # Apply thresholding to retain strongest areas and discard the rest
        _, sparse_thr = cv2.threshold(sparse_filtered,
                                      thresh=35,
                                      maxval=255,
                                      type=cv2.THRESH_TOZERO)

        # Discard areas where 2x2 structuring element will not fit
        sparse_opened = \
            img.grey_opening(sparse_thr, size=(2, 2)).astype(sparse_thr.dtype)

        # Segment using connected component labeling
        num_components, sparse_cc = \
            cv2.connectedComponents(sparse_opened, connectivity=4)
        sparse_cc = sparse_cc.astype(np.uint8)

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
        return num_components, sparse_cc, processing_stages

    def match_segments(self, frame, frame_prev, visual=False):
        """Analyze a pair of segmented frames and return conclusions about
        which segments match between frames."""
        # Only for first pass, when there is no other frame to compare to yet
        if frame_prev is None:
            frame_prev = np.zeros(frame.shape).astype(np.uint8)

        # Measure segment properties to use for likelihood matrix
        properties = measure.regionprops(frame)
        properties_prev = measure.regionprops(frame_prev)
        count = len(properties)
        count_prev = len(properties_prev)
        count_total = count + count_prev

        # Initialize coordinates as empty (fail case for no matches)
        coords = [[(0, 0) for col in range(2)] for row in range(count_total)]

        # Proceed only if there are segments in both frames
        if count and count_prev:
            # Initialize likelihood matrix
            likeilihood_matrix = np.zeros((count_total, count_total))

            # Matrix values: likelihood of segments being a match
            for seg_prev in properties_prev:
                for seg in properties:
                    # Convert segment labels to likelihood matrix indices
                    index_v = (seg_prev.label - 1)
                    index_h = (count_prev + seg.label - 1)

                    # Likeilihoods as a function of distance between segments
                    dist = distance.euclidean(seg_prev.centroid,
                                              seg.centroid)
                    # Map distance values using a Gaussian curve
                    likeilihood_matrix[index_v, index_h] = \
                        math.exp(-1 * (((dist - 10) ** 2) / 40))

            # Matrix values: likelihood of segments appearing/disappearing
            for i in range(count_total-1):
                # Compute closest distance from segment to edge of frame
                if i < count_prev:
                    point = properties_prev[i].centroid
                if count_prev <= i < (count + count_prev):
                    point = properties[i - count_prev].centroid
                edge_distance = min([point[0], point[1],
                                     self.height - point[0],
                                     self.width - point[1]])

                # Map distance values using an Exponential curve
                likeilihood_matrix[i, i] = (1 / 2) * math.exp(
                    -edge_distance / 10)

            # Convert likelihood matrix into cost matrix
            cost_matrix = -1*likeilihood_matrix
            cost_matrix -= cost_matrix.min()

            # Apply Hungarian/Munkres algorithm to find optimal matches
            _, matches = linear_sum_assignment(cost_matrix)

            # Convert matches (pairs of indices) into pairs of coordinates
            for i in range(count_total-1):
                j = matches[i]

                # Index condition if two segments are matching
                if (i < j) and (i < count_prev):
                    # Note: cv2.line requires int, .centroid returns float
                    float_coord1 = coords[i][0] = properties_prev[i].centroid
                    float_coord2 = properties[j - count_prev].centroid
                    coords[i][0] = tuple([int(val) for val in float_coord1])
                    coords[i][1] = tuple([int(val) for val in float_coord2])
                # Index condition for when a previous segment has disappeared
                if (i == j) and (i < count_prev):
                    float_coord1 = coords[i][0] = properties_prev[i].centroid
                    coords[i][0] = tuple([int(val) for val in float_coord1])
                # Index condition for when a new segment has appeared
                if (i == j) and (i >= count_prev):
                    float_coord2 = properties[j - count_prev].centroid
                    coords[i][1] = tuple([int(val) for val in float_coord2])

            # TODO: Write logic to convert pair coordinates into stats
            # e.g. "Appeared near chimney", "disappeared out of frame"

        # Proceed only if there are segments in either frame
        if count_total > 0:
            test = None

        if visual:
            # Scale labeled images to be visible with uint8 grayscale
            if count > 0:
                frame = frame*int(255/count)
            if count_prev > 0:
                frame_prev = frame_prev*int(255/count_prev)

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

        return coords, match_comparison


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
