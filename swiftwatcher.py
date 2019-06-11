import swiftwatcher.process_video as pv
import os
import argparse as ap
import csv

# Used for performance evaluation/grouth truth
import numpy as np


def save_test_details(params, count_estimate, load_directory, folder_name):
    """Save the full ground truth evaluation, as well as a summary of the test,
    to csv files.

    Some parameters include commas, so files are delimited with semicolons.

    Example:
        For a queue of size 21, pre-loading 10 frames is necessary to
    segment one frame (frame 11 is the first frame that can be segmented).
    Thus, for 100 frames loaded, only 90 will be segmented (and therefore only
    90 will have counts).

    Count labels:
        0, frame_number
        1, total_birds
        2, total_matches
        3, appeared_from_chimney
        4, appeared_from_edge
        5, appeared_ambiguous (could be removed, future-proofing for now)
        6, disappeared_to_chimney
        7, disappeared_to_edge
        8, disappeared_ambiguous (could be removed, future-proofing for now)
        9, outlier_behavior

    Estimate array contains a 10th catch-all count, "segmentation_error"."""

    # Create save directory if it does not already exist
    save_directory = load_directory + "/" + folder_name
    if not os.path.isdir(save_directory):
        try:
            os.makedirs(save_directory)
        except OSError:
            print("[!] Creation of the directory {0} failed."
                  .format(save_directory))

    # Comparing ground truth to estimated counts, frame by frame
    ground_truth = np.genfromtxt(load_directory + '/groundtruth.csv',
                                 delimiter=',').astype(dtype=int)
    num_counts = count_estimate.shape[0]
    results_full = np.hstack((ground_truth[0:num_counts, 0:10],
                              count_estimate[:, 0:10])).astype(np.int)

    # Using columns 1:10 so that frame numbers are excluded in error counting
    error_full = count_estimate[:, 1:10] - ground_truth[0:num_counts, 1:10]
    
    # Calculating when counts were overestimated
    error_over = np.copy(error_full)
    error_over[error_over < 0] = 0
    
    # Calculating when counts were underestimated
    error_under = np.copy(error_full)
    error_under[error_under > 0] = 0
    
    # Summarizing the performance of the algorithm across all frames
    results_summary = {
        "count_true": np.sum(ground_truth[0:num_counts, 1:10], axis=0),
        "count_estimated": np.sum(count_estimate[:, 1:10], axis=0),
        "error_net": np.sum(error_full, axis=0),
        "error_overestimate": np.sum(error_over, axis=0),
        "error_underestimate": np.sum(error_under, axis=0),
        "error_total": np.sum(abs(error_full), axis=0),
    }

    # Writing the full results to a file
    np.savetxt(save_directory+"/results_full.csv", results_full, delimiter=';')
    
    # Writing a summary of the parameters to a file
    with open(save_directory+"/parameters.csv", 'w') as csv_file:
        filewriter = csv.writer(csv_file, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(["PARAMETERS"])
        for key in params.keys():
            filewriter.writerow(["{}".format(key),
                                 "{}".format(params[key])])

    # Writing a summary of the results to a file
    with open(save_directory+"/results_summary.csv", 'w') as csv_file:
        filewriter = csv.writer(csv_file, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([" ", "SEGMNTS", "MATCHES",
                             "ENT_CHM", "ENT_FRM", "ENT_AMB",
                             "EXT_CHM", "EXT_FRM", "EXT_AMB", "OUTLIER"])
        for key in results_summary.keys():
            filewriter.writerow(["{}".format(key),
                                 "{}".format(results_summary[key][0]),
                                 "{}".format(results_summary[key][1]),
                                 "{}".format(results_summary[key][2]),
                                 "{}".format(results_summary[key][3]),
                                 "{}".format(results_summary[key][4]),
                                 "{}".format(results_summary[key][5]),
                                 "{}".format(results_summary[key][6]),
                                 "{}".format(results_summary[key][7]),
                                 "{}".format(results_summary[key][8])])


def main(args, params):
    """Count swift behavior (entering/exiting chimney) from video frames.

    To understand the current configuration of the algorithm, please look
    to the following functions, which are outside of main() below:

    - args: command-line arguments, used for file I/O, set by
        if __name__ == "__main__": block of code.
    - params: algorithm parameters, used to tweak processing stages, set by
        set_parameters() function."""

    # Code to extract all frames from video and save them to image files
    if args.extract:
        pv.extract_frames(args.video_dir, args.filename)

    # Code to process previously extracted frames
    else:
        # File I/O Initialization
        load_directory = (args.video_dir + os.path.splitext(args.filename)[0])

        # FrameQueue object (class for caching frames, processing frames)
        frame_queue = pv.FrameQueue(args.video_dir, args.filename,
                                    queue_size=params["queue_size"])
        frame_queue.stream.release()  # VideoCapture not needed for frame reuse
        frame_queue.frame_to_load_next = args.load[0]
        num_frames_to_analyse = args.load[1] - args.load[0]

        # Array to store estimated counts
        count_estimate = np.empty((0, 11), int)

        # The number of frames to read has an additional amount added,
        # "frame_queue.queue_center", because a cache of frames is needed to
        # segment a frame (equal to this amount). See pv.FrameQueue's
        # __init__() docstring for more information.
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
                count_estimate = np.vstack((count_estimate,
                                            match_counts)).astype(int)

            # Status updates
            if frame_queue.frames_read % 25 == 0:
                print("{0}/{1} frames processed."
                      .format(frame_queue.frames_read, num_frames_to_analyse))

            # Delay = 0 if fps == src_fps, delay > 0 if fps < src_fps
            frame_queue.frame_to_load_next += (1 + frame_queue.delay)

        save_test_details(params, count_estimate,
                          load_directory, args.custom_dir)


def set_parameters():
    """Dashboard for setting parameters for each processing stage of algorithm.

    Distinct from command line arguments. For this program, arguments are used
    for file I/O, directory selection, etc. These parameters affect the image
    processing and analysis parts of the algorithm instead."""

    params = {
        # Frame cropping
        "corners": [(760, 606), (920, 686)],  # (->, v), (--->, V)

        # Grayscale conversion
        "gs_algorithm": "cv2 default",

        # Robust PCA/motion estimation
        "queue_size": 21,
        "ialm_lmbda": 0.01,
        "ialm_tol": 0.001,
        "ialm_maxiter": 100,
        "ialm_darker": True,

        # Bilateral filtering
        "blf_iter": 2,     # How many times to iteratively apply
        "blf_diam": 7,     # Size of pixel neighbourhood
        "blf_sigma_c": 15,
        "blf_sigma_s": 1,

        # Thresholding
        "thr_type": 3,     # value of cv2.THRESH_TOZERO option
        "thr_value": 35,

        # Greyscale processing
        "gry_op_SE": (2, 2),

        # Labelled segmentation
        "seg_func": "cv2.connectedComponents(sparse_opened, "
                    "connectivity=4)",

        # Assignment Problem
        # Used to roughly map distanced into correct regions, but very hastily
        # done. Actual functions will be chosen much more methodically.
        "ap_func_match": "math.exp(-1 * (((dist - 10) ** 2) / 40))",
        "ap_func_notmatch": "(1 / 8) * math.exp(-edge_distance / 10)"
    }

    return params


if __name__ == "__main__":
    # Command line arguments used for specifying file I/O.
    # (NOT algorithm parameters. See set_parameters() for parameter choices.)
    parser = ap.ArgumentParser()
    parser.add_argument("-e",
                        "--extract",
                        help="Extract frames to HH:MM subfolders",
                        action="store_true"
                        )
    parser.add_argument("-l",
                        "--load",
                        help="Option to load previously saved frames",
                        nargs=2,
                        type=int,
                        metavar=('START_INDEX', 'END_INDEX'),
                        default=([7200, 16200])
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
                        default="tests/1_undecided-change"
                        )
    parser.add_argument("-v",
                        "--visual",
                        help="Output visualization of frame processing",
                        default=True
                        )
    arguments = parser.parse_args()

    parameters = set_parameters()
    main(arguments, parameters)
