import swiftwatcher.process_video as pv
import os
import argparse as ap
import math
import cv2
import numpy as np
import utils.cm as cm
from scipy import ndimage as img
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from skimage import measure

def main(args):
    # Code to extract all frames from video and save them to files
    if args.extract:
        pv.extract_frames(args.video_dir, args.filename)

    # Code to process previously extracted frames
    if args.reuse:
        # Initialize parameters necessary to load previously extracted frames
        # Directory follows assumed convention from pv.extract_frames()
        load_directory = (args.video_dir +
                          os.path.splitext(args.filename)[0])
        load_index = args.reuse[0]
        total_frames = args.reuse[1]

        # Create FrameQueue object
        queue_size = 20
        queue_center = int((queue_size-1)/2)
        frame_queue = pv.FrameQueue(args.video_dir, args.filename,
                                    queue_size=queue_size)
        frame_queue.stream.release()  # Not needed for frame reuse

        # Initialize data structures for bird counting stats
        bird_count = np.array([])
        properties_old = None
        sparse_frames_old = None
        sparse_cc_old = None
        ground_truth = np.genfromtxt('videos/groundtruth.csv',
                                     delimiter=',').astype(dtype=int)

        while frame_queue.frames_read < total_frames:
            # Load frame with specified index into FrameQueue object
            success = frame_queue.load_frame_from_file(load_directory,
                                                       load_index)

            # Process frame (grayscale, segmentation, etc.)
            if success:
                # Processing steps prior to motion estimation
                frame_queue.convert_grayscale()
                frame_queue.crop_frame(corners=args.crop)
                frame_queue.frame_to_column()

                if frame_queue.frames_read > queue_center:
                    # --------------- FRAME PROCESSING BEGINS --------------- #

                    # Choosing index such that RPCA will use adjacent frames
                    # (forward and backwards) to "queue_center" frame
                    lowrank, sparse = \
                        frame_queue.rpca_decomposition(index=queue_center,
                                                       darker_only=True)

                    # Apply bilateral filter to remove artifacts, retain birds
                    sparse_filtered = sparse
                    for i in range(2):
                        sparse_filtered = cv2.bilateralFilter(sparse_filtered,
                                                              d=7,
                                                              sigmaColor=15,
                                                              sigmaSpace=1)

                    # Retain strongest areas and discard the rest
                    _, sparse_thr = cv2.threshold(sparse_filtered,
                                                  thresh=35,
                                                  maxval=255,
                                                  type=cv2.THRESH_TOZERO)

                    sparse_opened = \
                        img.grey_opening(sparse_thr, size=(2, 2)) \
                        .astype(sparse_thr.dtype)

                    # Segment using connected component labeling
                    retval, sparse_cc = \
                        cv2.connectedComponents(sparse_opened, connectivity=4)
                    # Scale CC image for visual clarity

                    # ------ PROCESSING ENDS, STATS MEASUREMENT BEGINS ------ #

                    properties_new = measure.regionprops(sparse_cc)
                    if properties_old is None:
                        properties_old = properties_new

                    count_old = len(properties_old)
                    count_new = len(properties_new)
                    cost_matrix = np.zeros((count_new + count_old,
                                            count_new + count_old))

                    # Filling in cost matrix for object pairs
                    for seg_old in properties_old:
                        for seg_new in properties_new:
                            index_v = (seg_old.label-1)
                            index_h = (count_old+seg_new.label-1)

                            # Initial cost function as inversely proportional
                            # to the distance between object pairs
                            cost_matrix[index_v, index_h] = \
                                1/(0.0001 +
                                   distance.euclidean(seg_old.centroid,
                                                      seg_new.centroid))*100

                    # Filling in cost matrix for "appear"/"disappear"
                    for i in range(count_new + count_old):
                        if i < count_old:
                            coord = properties_old[i].centroid
                        if count_old <= i < (count_new + count_old):
                            coord = properties_new[i-count_old].centroid

                        edge_proximity = min([coord[0],
                                              coord[1],
                                              frame_queue.height-coord[0],
                                              frame_queue.width-coord[1]])

                        # Exponential function returns 1 when edge proximity
                        # is near zero, but drops off when edge proximity is
                        # large.
                        cost_matrix[i, i] = math.exp(-edge_proximity/10)

                    if count_new and count_old:
                        # Turn maximization problem into minimization
                        cost_matrix_min = -1 * cost_matrix
                        cost_matrix_min -= cost_matrix_min.min()
                        _, assignments = \
                            linear_sum_assignment(cost_matrix_min)

                        # Convert bird pairs into coordinate pairs
                        assignment_coords = \
                            np.zeros((len(assignments), 4)).astype(np.int)
                        for i in range(len(assignments)):
                            # 'i' is label of entry in likelihood matrix
                            # 'j' is label of corresponding assignment
                            j = assignments[i]
                            if i < count_old:
                                assignment_coords[i, 0:2] = \
                                    properties_old[i].centroid
                                if i < j:
                                    assignment_coords[i, None, 2:4] = \
                                        properties_new[j-count_old].centroid
                            if i >= count_old:
                                if i == j:
                                    assignment_coords[i, None, 2:4] = \
                                        properties_new[j-count_old].centroid

                        # Total count for foreground segments only
                        bird_count = np.append(bird_count, count_new)

                    properties_old = properties_new

                    # ----- STATS MEASUREMENT ENDS, SAVE TO FILE BEGINS ----- #

                    # Modify frame stages for visual clarity only
                    frame = np.reshape(frame_queue.queue[queue_center],
                                       (frame_queue.height, frame_queue.width))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, '{}'.format(save_index), (0, 12),
                                fontFace=font, fontScale=0.5,
                                color=(255, 255, 255))

                    if retval > 0:
                        sparse_cc_new = sparse_cc*(255/retval)
                    else:
                        sparse_cc_new = sparse_cc

                    # Compare different processing stages for single frame
                    separator_v = 64*np.ones(shape=(1, frame_queue.width),
                                             dtype=np.uint8)
                    sparse_frames_new = np.vstack((frame, separator_v,
                                                   sparse, separator_v,
                                                   sparse_filtered, separator_v,
                                                   sparse_thr, separator_v,
                                                   sparse_opened, separator_v,
                                                   sparse_cc_new))

                    # Compare processing stages for two frames side-by-side
                    if sparse_frames_old is None:
                        sparse_frames_old = np.zeros(sparse_frames_new.shape)
                    separator_h = 255*np.ones(shape=
                                              (sparse_frames_new.shape[0], 1),
                                              dtype=np.uint8)
                    fr_comparison = np.hstack((sparse_frames_old, separator_h,
                                               sparse_frames_new))
                    sparse_frames_old = sparse_frames_new

                    # Compare segmented images
                    if sparse_cc_old is None:
                        sparse_cc_old = np.zeros(sparse_cc_new.shape)
                    seg_comparison = np.vstack((sparse_cc_old, separator_v,
                                                sparse_cc_new))

                    for i in range(count_old):
                        p1o = (assignment_coords[i, 1],
                               assignment_coords[i, 0])
                        p2o = (assignment_coords[i, 3],
                               assignment_coords[i, 2]+frame_queue.height+1)
                        color = (255, 255, 255)
                        test1 = np.count_nonzero(p1o)
                        test2 = np.count_nonzero(p2o)
                        if (np.count_nonzero(p1o)+np.count_nonzero(p2o)) == 4:
                            cv2.line(seg_comparison, p1o, p2o,
                                     color, thickness=1)

                    # TODO: Draw lines between coords
                    frame_queue.save_frame_to_file(load_directory,
                                                   frame=seg_comparison,
                                                   index=queue_center,
                                                   folder_name=args.custom_dir,
                                                   scale=400)
                    sparse_cc_old = sparse_cc_new

                    # Save comparison to file for viewing convenience
                    # frame_queue.save_frame_to_file(load_directory,
                    #                                frame=fr_comparison,
                    #                                index=queue_center,
                    #                                folder_name=args.custom_dir,
                    #                                scale=400)

                    # ------------------ SAVE TO FILE ENDS -------------------#

            load_index += (1 + frame_queue.delay)
            save_index = load_index - queue_center
            if frame_queue.frames_read % 50 == 0:
                print("{0}/{1} frames processed."
                      .format(frame_queue.frames_read, total_frames))

        # Calculate error for total bird counts
        ground_truth = np.c_[ground_truth,
                             bird_count.reshape(-1, 1).astype(np.int)]
        error = ground_truth[:, 1] - ground_truth[:, 6]
        error_less = sum(error[error > 0])
        error_more = -1* sum(error[error < 0])

    breakpoint()


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-e",
                        "--extract",
                        help="Extract frames to HH:MM subfolders",
                        action="store_true"
                        )
    parser.add_argument("-r",
                        "--reuse",
                        help="Option to reuse previously saved frames",
                        nargs=2,
                        type=int,
                        metavar=('START_FRAME', 'TOTAL_FRAMES'),
                        default=([16150, 300])
                        )
    parser.add_argument("-d",
                        "--video_dir",
                        help="Path to directory containing video file",
                        default="videos/"
                        )
    parser.add_argument("-f",
                        "--filename",
                        help="Name of video file",
                        default="ch04_20170518205849.mp4"
                        )
    parser.add_argument("-c",
                        "--custom_dir",
                        help="Custom directory for extracted frame files",
                        default="Test Folder"
                        )
    parser.add_argument("-p",
                        "--crop",
                        help="Corner coordinates for cropping.",
                        default=[(760, 650), (921, 686)]
                        )
    arguments = parser.parse_args()

    main(arguments)
