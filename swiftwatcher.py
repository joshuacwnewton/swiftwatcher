# ================================= TODOs =================================== #
# Improving segmentation:
# -TODO: Add MSER scratch code to secondary segmentation function
#
# Automating tests for different parameters:
# -TODO: Research differences between JSON files and other storage types
# -TODO: Save algorithm parameters to a JSON file
# -TODO: Write code to load algorithm parameters from a JSON file
#
# Transition to pandas:
# -TODO: Convert save_test_results() into using dataframes.
#
# File I/O:
# -TODO: Look at how program handles "default directory" formatting
#
# Minor refactoring:
# -TODO: PEP8 refactoring (always!)
# =========================================================================== #

import swiftwatcher.video_processing as vid
import swiftwatcher.data_analysis as data
import argparse as ap
import os


def main(args, params):
    """To understand the current configuration of the algorithm, please look
    to the following functions, which are outside of main() below:

    - args: command-line arguments, used for file I/O, set by
        if __name__ == "__main__": block of code.
    - params: algorithm parameters, used to tweak processing stages, set by
        set_parameters() function."""

    if args.extract:
        vid.extract_frames(args)

    else:
        data.save_test_config(args, params)

        df_estimation = vid.process_extracted_frames(args, params)
        # It's called "df_estimation" because save_test_results converts from
        # dataframe to ndarray for parsing. When save_test_results() gets
        # rewritten to use dataframes, the "df_" prefix should be dropped

        data.save_test_results(args, df_estimation)

        # Generate cumulative sums and compare for ground truth + estimation
        # data.plot_function_for_testing(args, df_estimate, df_groundtruth)


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
        "thr_value": 50,

        # Greyscale processing
        "gry_op_SE": (2, 2),

        # Labelled segmentation
        "seg_func": "cv2.connectedComponents(sparse_opened, "
                    "connectivity=4)",

        # Assignment Problem
        # Used to roughly map distances into correct regions, but very hastily
        # done. Actual functions will be chosen much more methodically.
        "ap_func_match": "math.exp(-1 * (((dist - 10) ** 2) / 40))",
        "ap_func_notmatch": "(1 / 8) * math.exp(-edge_distance / 10)"
    }

    return params


if __name__ == "__main__":
    # Command line arguments used for specifying file I/O.
    # (NOT algorithm parameters. See set_parameters() for parameter choices.)
    parser = ap.ArgumentParser()

    # Flag to determine which mode to run the program in, <EXTRACT/LOAD>
    # (if flag is not provided, previously extracted frames will be loaded)
    parser.add_argument("-e",
                        "--extract",
                        help="Extract frames to HH:MM subfolders",
                        action="store_true"
                        )

    # General arguments for video file I/O
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
    parser.add_argument("-t",
                        "--timestamp",
                        help="In-frame timestamp for start of video",
                        default="2017-05-18 20:58:49.000000000"
                        )

    # Arguments for running image processing/analysis tests
    parser.add_argument("-l",
                        "--load",
                        help="Specify indices to load previously saved frames",
                        nargs=2,
                        type=int,
                        metavar=('START_INDEX', 'END_INDEX'),
                        default=([7200, 7250])
                        )
    parser.add_argument("-c",
                        "--custom_dir",
                        help="Custom directory for saving various things",
                        default="/tests/refactor_dataframe"
                        )
    parser.add_argument("-v",
                        "--visual",
                        help="Output visualization of frame processing",
                        default=True
                        )
    parser.add_argument("-g",
                        "--groundtruth",
                        help="Path to ground truth file",
                        default="/groundtruth/groundtruth.csv"
                        )
    arguments = parser.parse_args()

    # Repeatedly used path. Storing here because it is a derived from only
    # arguments, and it makes more sense than to repeatedly derive it.
    arguments.load_dir = (arguments.video_dir +
                          os.path.splitext(arguments.filename)[0])

    parameters = set_parameters()
    main(arguments, parameters)
