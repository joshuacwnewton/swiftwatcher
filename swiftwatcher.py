import swiftwatcher.process_video as pv
import os
import argparse as ap
import csv

# Used for performance evaluation/grouth truth
import numpy as np


def save_test_details(params, count_estimate, load_directory, folder_name):
    """Save the full ground truth comparison, as well as a summary of the test,
    to csv files. Note, some parameters include commas, so .csv files are
    delimited with semicolons instead of commas."""

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
    results_full = np.hstack((ground_truth, count_estimate[:, 1:6])) \
        .astype(np.int)
    error_full = count_estimate[:, 1:6] - ground_truth[:, 1:6]
    
    # Calculating when counts were overestimated
    error_over = np.copy(error_full)
    error_over[error_over < 0] = 0
    
    # Calculating when counts were underestimated
    error_under = np.copy(error_full)
    error_under[error_under > 0] = 0
    
    # Summarizing the performance of the algorithm across all frames
    results_summary = {
        "count_true": np.sum(ground_truth[:, 1:6], axis=0),
        "count_estimated": np.sum(count_estimate[:, 1:6], axis=0),
        "error_net": np.sum(error_full, axis=0),
        "error_overestimate": np.sum(error_over, axis=0),
        "error_underestimate": np.sum(error_under, axis=0),
        "error_total": np.sum(abs(error_full), axis=0),
    }

    # Writing the full results to a file
    np.savetxt(save_directory+'/results_full.csv', results_full, delimiter=";")
    
    # Writing a summary of the parameters to a file
    with open(save_directory+'/parameters.csv', 'w') as csv_file:
        filewriter = csv.writer(csv_file, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(["PARAMETERS"])
        for key in params.keys():
            filewriter.writerow(["{}".format(key),
                                 "{}".format(params[key])])

    # Writing a summary of the results to a file
    with open(save_directory+'/results_summary.csv', 'w') as csv_file:
        filewriter = csv.writer(csv_file, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([" ", "TOTAL BIRDS/SEGMENTS",
                             "ENTERED CHIMNEY", "ENTERED FRAME",
                             "EXITED CHIMNEY", "EXITED FRAME"])
        for key in results_summary.keys():
            filewriter.writerow(["{}".format(key),
                                 "{}".format(results_summary[key][0]),
                                 "{}".format(results_summary[key][1]),
                                 "{}".format(results_summary[key][2]),
                                 "{}".format(results_summary[key][3]),
                                 "{}".format(results_summary[key][4])])


def main(args, params):
    """Count swift behavior (entering/exiting chimney) from video frames.

    To understand the current configuration of the algorithm, please look
    to the following functions, which are outside of main() below:

    - args: command-line arguments, used for file I/O, set by
        if __name__ == "__main__": block of code.
    - params: algorithm parameters, used to tweak processing stages, set by
        set_parameters() function."""

    # TODO: Switch from "extract frames -> reload frames" to real-time analysis
    # Written this way to save time when testing with specific time ranges

    # Code to extract all frames from video and save them to image files
    if args.extract:
        pv.extract_frames(args.video_dir, args.filename)

    # Code to process previously extracted frames
    if args.load:
        # --------------------- INITIALIZATION BEGINS ----------------------- #

        # File I/O
        load_directory = (args.video_dir + os.path.splitext(args.filename)[0])
        frame_to_load = args.load[0]
        num_frames_to_read = args.load[1] - args.load[0]

        # FrameQueue object (class for caching frames, processing frames)
        frame_queue = pv.FrameQueue(args.video_dir, args.filename,
                                    queue_size=params["queue_size"])
        frame_queue.stream.release()  # VideoCapture not needed for frame reuse

        # Array to store estimated counts
        count_estimate = np.array([]).reshape(0, 8)

        # -------- INITIALIZATION ENDS, PROCESSING ALGORITHM BEGINS --------- #

        while frame_queue.frames_read < num_frames_to_read:
            # Load frame into index 0 and apply preprocessing
            frame_queue.load_frame_from_file(load_directory, frame_to_load)
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

            # Save counts (only for frames with ground truth)
            if 16200 <= (frame_to_load-frame_queue.queue_center) <= 16390:
                count_estimate = np.vstack((count_estimate,
                                            match_counts)).astype(int)

            # Status updates
            if frame_queue.frames_read % 25 == 0:
                print("{0}/{1} frames processed."
                      .format(frame_queue.frames_read, num_frames_to_read))

            # Delay = 0 if fps == src_fps, delay > 0 if fps < src_fps
            frame_to_load += (1 + frame_queue.delay)

        # ---------- PROCESSING ALGORITHM ENDS, TEST OUTPUT BEGINS ---------- #

        save_test_details(params, count_estimate,
                          load_directory, args.custom_dir)


def set_parameters():
    """Dashboard for setting parameters for each processing stage of algorithm.

    Distinct from command line arguments. For this program, arguments are used
    for file I/O, directory selection, etc. These parameters affect the image
    processing and analysis parts of the algorithm instead."""
    params = {
        # Frame cropping
        "corners": [(760, 650), (921, 686)],

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
        "ap_func_notmatch": "(1 / 2) * math.exp(-edge_distance / 10)"
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
                        default=([16150, 16450])
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
                        default="tests/refactor check"
                        )
    parser.add_argument("-v",
                        "--visual",
                        help="Output visualization of frame processing",
                        default=True
                        )
    arguments = parser.parse_args()

    parameters = set_parameters()
    main(arguments, parameters)
