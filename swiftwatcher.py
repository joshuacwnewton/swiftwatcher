import swiftwatcher.process_video as pv
import os
import argparse as ap
import csv

# Used for performance evaluation/grouth truth
import numpy as np

# TODO: Switch from "extract frames -> reload frames" to real-time analysis
# Written this way to save time when testing with specific timestamps


def save_test_details(params, stats_array, load_directory, folder_name):
    ground_truth = np.genfromtxt(load_directory + '/groundtruth.csv',
                                 delimiter=',').astype(dtype=int)

    full_results = np.hstack((ground_truth, stats_array[:, 1:6])) \
        .astype(np.int)
    error = stats_array[:, 1:6] - ground_truth[:, 1:6]
    error_over = np.copy(error)
    error_over[error_over < 0] = 0
    error_under = np.copy(error)
    error_under[error_under > 0] = 0
    performance = {
        # [TOTAL_SEGM][ENTER_CHIMN][ENTER_FRAME][EXIT_CHIMN][EXIT_FRAME])
        # "error_all": error,
        "error_abs": np.sum(abs(error), axis=0),
        "error_net": np.sum(error, axis=0),
        "error_over": np.sum(error_over, axis=0),
        "error_under": np.sum(error_under, axis=0),
        "count_true": np.sum(ground_truth[:, 1:6], axis=0),
        "count_estimated": np.sum(stats_array[:, 1:6], axis=0),
        # "count_all": np.hstack((ground_truth[:, 1:6],
        # stats_array[:, 1:6]))
    }

    np.savetxt(load_directory + '/' + folder_name + '/a_full-results.csv',
               full_results, delimiter=",")
    with open(load_directory + '/' + folder_name +
              '/b_test_summary.csv', 'w') as csv_file:
        filewriter = csv.writer(csv_file, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(["PARAMETERS"])
        for key in params.keys():
            filewriter.writerow(["{}".format(key),
                                 "{}".format(params[key])])
        filewriter.writerow(["RESULTS"])
        for key in performance.keys():
            filewriter.writerow(["{}".format(key),
                                 "{}".format(performance[key])])


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
    if args.load:
        # --------------------- INITIALIZATION BEGINS ----------------------- #

        # File I/O
        load_directory = (args.video_dir + os.path.splitext(args.filename)[0])
        save_directory = args.custom_dir + "/frames"
        frame_to_load = args.load[0]
        num_frames_to_read = args.load[1] - args.load[0]

        # FrameQueue object (class for caching frames, processing frames)
        frame_queue = pv.FrameQueue(args.video_dir, args.filename,
                                    queue_size=params["queue_size"])
        frame_queue.stream.release()  # Videocapture not needed for frame reuse

        # Ground truth and measured stats
        stats_array = np.array([]).reshape(0, 6)

        # -------- INITIALIZATION ENDS, PROCESSING ALGORITHM BEGINS --------- #

        while frame_queue.frames_read < num_frames_to_read:
            # Load frame into index 0 and apply preprocessing
            frame_queue.load_frame_from_file(load_directory, frame_to_load)
            frame_queue.convert_grayscale(algorithm=params["gs_algorithm"])
            frame_queue.crop_frame(corners=params["corners"])
            frame_queue.frame_to_column()

            # Proceed only when enough frames are stored for motion estimation
            if frame_queue.frames_read > frame_queue.queue_center:
                # Segment frame using context from adjacent frames in queue
                processing_stages = \
                    frame_queue.segment_frame(
                        lmbda=params["ialm_lmbda"],
                        tol=params["ialm_tol"],
                        maxiter=params["ialm_maxiter"],
                        darker=params["ialm_darker"],
                        iters=params["blf_iter"],
                        diameter=params["blf_diam"],
                        sigma_space=params["blf_sigma_s"],
                        sigma_color=params["blf_sigma_c"],
                        thr_value=params["thr_value"],
                        thr_type=params["thr_type"],
                        gry_op_SE=params["gry_op_SE"],
                        segmentation=params["seg_func"],
                        index=frame_queue.queue_center,
                        visual=True
                    )

                # Match bird segments from two sequential frames.
                match_coords, match_stats, match_comparison = \
                    frame_queue.match_segments(
                        match_function=params["ap_func_match"],
                        notmatch_function=params["ap_func_notmatch"],
                        visual=True
                    )

                # Save results to image files for visual inspection.
                if processing_stages is not None:
                    frame_queue.save_frame_to_file(load_directory,
                                                   frame=processing_stages,
                                                   index=frame_queue.queue_center,
                                                   folder_name=save_directory,
                                                   file_prefix="seg_",
                                                   scale=400)
                if match_comparison is not None:
                    frame_queue.save_frame_to_file(load_directory,
                                                   frame=match_comparison,
                                                   index=frame_queue.queue_center,
                                                   folder_name=save_directory,
                                                   scale=400)

                # Store relevant matching stats in ground_truth data structure.
                if 16200 <= (frame_to_load-frame_queue.queue_center) <= 16390:
                    total_birds = (match_stats["total_matches"] +
                                   match_stats["appeared_from_edge"] +
                                   match_stats["appeared_from_chimney"])
                    stats_list = [frame_to_load-frame_queue.queue_center,
                                  total_birds,
                                  match_stats["appeared_from_chimney"],
                                  match_stats["appeared_from_edge"],
                                  match_stats["disappeared_to_chimney"],
                                  match_stats["disappeared_to_edge"]]
                    stats_array = np.vstack((stats_array,
                                             stats_list)).astype(int)

            frame_to_load += (1 + frame_queue.delay)
            if frame_queue.frames_read % 25 == 0:
                print("{0}/{1} frames processed."
                      .format(frame_queue.frames_read, num_frames_to_read))

        # ----------- FRAME PROCESSING ENDS, STATS OUTPUT BEGINS ------------ #

        save_test_details(params, stats_array, load_directory, args.custom_dir)


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
        "gry_op_SE": (2, 2),  # Structuring element for greyscale opening

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
                        default="tests/directory test"
                        )
    arguments = parser.parse_args()

    parameters = set_parameters()
    main(arguments, parameters)
