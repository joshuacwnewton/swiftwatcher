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

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn; seaborn.set()


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

    def __init__(self, video_directory, filename, queue_size=1,
                 desired_fps=False):
        # Check validity of filepath
        video_filepath = video_directory + filename
        if not os.path.isfile(video_filepath):
            raise Exception("[!] Filepath does not point to valid video file.")

        # Open source video file and initialize its immutable attributes
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
        else:
            self.fps = desired_fps
        # Delay parameter needed when subsampling video
        self.delay = round(self.src_fps / self.fps) - 1
        self.frame_to_load_next = 0

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

        # Load frame from file ([:11] -> limit precision to get things to work
        # because I increased precision after extracting frames)
        # TODO: Remove this precision limit when frames are re-extracted
        file_paths = glob.glob("{0}/frame{1}_{2}*".format(save_directory,
                                                          frame_number,
                                                          timestamp[:11]))
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
                           base_folder="", frame_folder="/frames",
                           file_prefix="", file_suffix=""):
        """Save an individual frame to an image file. If frame itself is not
        provided, frame will be pulled from frame_queue at specified index."""

        # By default, frames will be saved in a subfolder corresponding to
        # HH:MM formatting. However, a custom subfolder can be chosen instead.
        base_save_directory = base_save_directory+base_folder+frame_folder

        if single_folder:
            save_directory = base_save_directory
        else:
            time = self.timestamps[index].split(":")
            save_directory = base_save_directory+"/"+time[0]+":"+time[1]

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
                                self.timestamps[index],
                                file_suffix),
                        frame)
        except Exception as e:
            print("[!] Image saving failed due to: {0}".format(str(e)))

    def convert_grayscale(self, index=0, algorithm="cv2 default"):
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

    def segment_frame(self, load_directory, folder_name,
                      params, visual=False):
        """Segment birds from one frame ("index") using information from other
        frames in the FrameQueue object. Store segmented frame in secondary
        queue."""

        # Set segmented image to empty if not enough frames have been read
        # to do motion analysis.
        if (self.frames_read-1) < self.queue_center:
            sparse_cc = np.zeros((self.height, self.width), dtype=np.int)
        else:
            # Apply Robust PCA method to isolate regions of motion
            # lowrank = "background" image
            # sparse  = "foreground" errors corrupting the "background" image
            # frame = lowrank + sparse
            lowrank, sparse = self.rpca(params["ialm_lmbda"],
                                        params["ialm_tol"],
                                        params["ialm_maxiter"],
                                        params["ialm_darker"],
                                        index=self.queue_center)

            if self.framenumbers[self.queue_center] == 12954:
                mser = cv2.MSER_create(_min_area=6, _max_area=1000)
                regions, _ = mser.detectRegions(cv2.bitwise_not(sparse))
                vis = sparse.copy()
                hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
                mask = np.zeros((sparse.shape[0], sparse.shape[1], 1),
                                dtype=np.uint8)
                mask = cv2.dilate(mask, np.ones((150, 150), np.uint8))
                count = 1
                for contour in hulls:
                    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
                    self.save_frame_to_file(load_directory,
                                            frame=mask,
                                            index=self.queue_center,
                                            base_folder=folder_name,
                                            frame_folder="/mser",
                                            file_suffix=str(count),
                                            scale=4)
                    count += 1
                test = None

            # Apply bilateral filter to smooth over low-contrast regions
            sparse_filtered = sparse
            for i in range(params["blf_iter"]):
                sparse_filtered = cv2.bilateralFilter(sparse_filtered,
                                                      params["blf_diam"],
                                                      params["blf_sigma_s"],
                                                      params["blf_sigma_c"])

            # Apply thresholding to retain strongest areas and discard the rest
            _, sparse_thr = cv2.threshold(sparse_filtered,
                                          thresh=params["thr_value"],
                                          maxval=255,
                                          type=params["thr_type"])

            # Discard areas where 2x2 structuring element will not fit
            sparse_opened = \
                img.grey_opening(sparse_thr, size=params["gry_op_SE"]) \
                .astype(sparse_thr.dtype)

            # Segment using connected component labeling
            num_components, sparse_cc = eval(params["seg_func"])
            # num_components, sparse_cc = \
            #     cv2.connectedComponents(sparse_opened, connectivity=conn)

            # Create visualization of processing stages if requested
            if visual:
                # Fetch unprocessed frame, reshape into image
                frame = np.reshape(self.queue[self.queue_center],
                                   (self.height, self.width))

                # Scale labeled image to be visible with uint8 grayscale
                if num_components > 0:
                    sparse_cc_scaled = sparse_cc*int(255/num_components)
                else:
                    sparse_cc_scaled = sparse_cc

                # Combine stages into one image, separated for visual clarity
                separator = 64*np.ones(shape=(1, self.width), dtype=np.uint8)
                processing_stages = np.vstack((frame, separator,
                                               sparse, separator,
                                               sparse_filtered, separator,
                                               sparse_thr, separator,
                                               sparse_opened, separator,
                                               sparse_cc_scaled)
                                              ).astype(np.uint8)

                # Save to file
                self.save_frame_to_file(load_directory,
                                        frame=processing_stages,
                                        index=self.queue_center,
                                        base_folder=folder_name,
                                        frame_folder="/segmentation",
                                        scale=4)

        # Append empty values first if queue is empty
        if self.seg_queue.__len__() is 0:
            self.seg_queue.appendleft(np.zeros((self.height, self.width))
                                      .astype(np.uint8))
            self.seg_properties.appendleft([])

        # Append segmented frame (and information about frame) to queue
        self.seg_queue.appendleft(sparse_cc.astype(np.uint8))
        self.seg_properties.appendleft(measure.regionprops(sparse_cc))

    def match_segments(self, load_directory, folder_name,
                       params, visual=False):
        """Analyze a pair of segmented frames and return conclusions about
        which segments match between frames."""
        
        # Assign name to commonly used properties
        count = len(self.seg_properties[0])
        count_prev = len(self.seg_properties[1])
        count_total = count + count_prev

        # Initialize coordinates and counts
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
                    chimney_distance = self.height - coord_pair[1][0]

                    if edge_distance <= 10:
                        counts["ENT_FRM"] += 1
                    elif chimney_distance <= 10:
                        counts["ENT_CHM"] += 1
                    else:
                        counts["SEG_ERR"] += 1
                        frame_err += 1
                # If this condition is met, pair means a segment disappeared
                elif coord_pair[1] == (0, 0):
                    edge_distance = min(coord_pair[0][0], coord_pair[0][1],
                                        self.width - coord_pair[0][1])
                    chimney_distance = self.height - coord_pair[0][0]

                    if edge_distance <= 10:
                        counts["EXT_FRM"] += 1
                    elif chimney_distance <= 10:
                        counts["EXT_CHM"] += 1
                    else:
                        counts["SEG_ERR"] += 1
                        frame_prev_err += 1
                # Otherwise, a match was made
                else:
                    counts["MATCHES"] += 1

        # Create visualization of segment matches if requested
        if visual:
            frame = np.copy(self.seg_queue[0])
            frame_prev = np.copy(self.seg_queue[1])

            # Colormappings for tab20 colormap in colormap
            colors = [14, 40, 118, 144, 170, 222, 248,
                      1, 27, 105, 131, 157, 209, 235]  # non-G/R colors
            appeared_color = 53     # Green
            disappeared_color = 79  # Red

            # Color matching between frames (grayscale, colormap comes later)
            frame_prev[frame_prev == 0] = 196
            frame[frame == 0] = 196
            color_index = 0
            for i in range(count_total):
                j = matches[i]
                # Index condition if two segments are matching
                if (i < j) and (i < count_prev):
                    frame_prev[frame_prev == (i+1)] = colors[color_index]
                    frame[frame == (j+1 - count_prev)] = colors[color_index]
                    color_index += 1
                # Index condition for when a previous segment has disappeared
                elif (i == j) and (i < count_prev):
                    frame_prev[frame_prev == (i+1)] = disappeared_color
                # Index condition for when a new segment has appeared
                elif (i == j) and (i >= count_prev):
                    frame[frame == (j+1 - count_prev)] = appeared_color

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
            separator_v = 183*np.ones(shape=(self.height*scale, 1),
                                      dtype=np.uint8)
            match_comparison = np.hstack((frame_prev, separator_v, frame))

            # Adding horizontal bar to display frame information
            horizontal_bg = 183*np.ones(shape=(50, match_comparison.shape[1]),
                                        dtype=np.uint8)
            match_comparison = np.vstack((match_comparison, horizontal_bg))

            # Write text on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(match_comparison,
                        'Frame {0} -- Exit, edge: {1} | Exit, chimn: {2} | '
                        'False positive: {3}              '
                        'Frame {4} -- Enter, edge: {5} | Enter, chimn: {6} | '
                        'False positive: {7}'.format(counts["FRM_NUM"]-1,
                                                     counts["EXT_FRM"],
                                                     counts["EXT_CHM"],
                                                     frame_prev_err,
                                                     counts["FRM_NUM"],
                                                     counts["ENT_FRM"],
                                                     counts["ENT_CHM"],
                                                     frame_err),
                        (10, 350), font, 0.5, 196, 2)

            # Apply color mapping
            match_comparison_color = cm.apply_custom_colormap(match_comparison,
                                                              cmap="tab20")

            # Save image to folder
            self.save_frame_to_file(load_directory,
                                    frame=match_comparison_color,
                                    index=self.queue_center,
                                    base_folder=folder_name,
                                    frame_folder="/matches",
                                    scale=1)

        return counts


def process_extracted_frames(args, params):
    """Function which uses class methods to analyse a sequence of previously
    extracted frames, and determine bird counts for that sequence."""

    # Load direction matches formatting found in extract_frames()
    load_directory = args.video_dir + os.path.splitext(args.filename)[0]

    # FrameQueue object (class for caching frames and applying
    #                    image processing to cached frames)
    frame_queue = FrameQueue(args.video_dir, args.filename,
                             queue_size=params["queue_size"])
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
    while frame_queue.frames_read < (num_frames_to_analyse +
                                     frame_queue.queue_center):

        # Load frame into index 0 and apply preprocessing
        frame_queue.load_frame_from_file(load_directory,
                                         frame_queue.frame_to_load_next)
        frame_queue.convert_grayscale(algorithm=params["gs_algorithm"])
        frame_queue.crop_frame(corners=params["corners"])
        frame_queue.frame_to_column()

        # Proceed only when enough frames are stored for motion estimation
        if frame_queue.frames_read > frame_queue.queue_center:
            frame_queue.segment_frame(load_directory,
                                      args.custom_dir,
                                      params,
                                      visual=args.visual)
            match_counts = frame_queue.match_segments(load_directory,
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

    return count_estimate


def extract_frames(video_directory, filename, queue_size=1,
                   save_directory=None):
    """Function which uses class methods to extract individual frames
     (one at a time) from a video file. Saves each frame to image files for
     future reuse."""

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


def ms_to_timestamp(total_ms):
    """Helper function to convert millisecond value into timestamp."""
    # cv2's VideoCapture class provides the position of the video in
    # milliseconds (cv2.CAP_PROP_POS_MSEC). However, as it can be easier to
    # think of video in terms of hours, minutes, and seconds, it's helpful to
    # be able to convert back and forth.
    total_s = int(total_ms/1000)
    total_m = int(total_s/60)
    total_h = int(total_m/60)

    ns = round(1000000*(total_ms % 1000))
    s = round(total_s % 60)
    m = round(total_m % 60)
    h = round(total_h % 24)

    timestamp = "{0:02d}:{1:02d}:{2:02d}:{3:09d}".format(h, m, s, ns)
    return timestamp


def timestamp_to_ms(timestamp):
    """Helper function to convert timestamps to millisecond totals."""
    # cv2's VideoCapture class provides the position of the video in
    # milliseconds (cv2.CAP_PROP_POS_MSEC). However, as it can be easier to
    # think of video in terms of hours, minutes, and seconds, it's helpful to
    # be able to convert back and forth.
    time = timestamp.split(":")
    total_ms = (float(time[0])*60*60*1000 +
                float(time[1])*60*1000 +
                float(time[2])*1000 +
                float(time[3]))

    return total_ms


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
