# =========================================================================== #
# Note: Each category of TODOs represents a git branch.
# =========================================================================== #
# algorithm-improvements:
# -TODO: Test smaller crop-regions for RPCA speed increase.
# -TODO: Test smaller scaled image (pyramid-down?) for RPCA speed increase.
# -TODO: Test smaller frame_queue for RPCA speed increase.
#
# test-automation:
# -TODO: Read up on differences between JSON files and other storage types.
# -TODO: Save algorithm parameters to a JSON file.
# -TODO: Load algorithm parameters from a JSON file.
# -TODO: Save command line arguments to JSON file.
# -TODO: Load command line arguments from JSON file.
#
# algorithm-visualization:
#
# results-visualization:
#
# results-csv-files:
# -TODO: Update empty-gt-generator for us timestamp precision.
#
# running tests:
#
# =========================================================================== #

import swiftwatcher.video_processing as vid
import swiftwatcher.data_analysis as data
import pandas as pd
import argparse as ap
import os
import time


def main(args, params):
    """To understand the current configuration of the algorithm, please look
    to the following functions, which are outside of main() below:

    - args: command-line arguments, used for file I/O, set by
        if __name__ == "__main__": block of code.
    - params: algorithm parameters, used to tweak processing stages, set by
        set_parameters() function."""

    if args.extract:
        vid.extract_frames(args)
        pass
    if args.process:
        start = time.time()
        df_eventinfo = vid.process_frames(args, params)
        end = time.time()

        elapsed_time = pd.to_timedelta((end - start), 's')
        print("[-] Frame processing took {}.".format(elapsed_time))

        data.save_test_config(args, params)
    if args.analyse:
        if args.process:
            dfs = data.import_dataframes(args, ["groundtruth"])
            dfs["eventinfo"] = df_eventinfo
            dfs["features"] = data.generate_feature_vectors(dfs["eventinfo"])
            dfs["prediction"] = data.classify_feature_vectors(dfs["features"])
            data.export_dataframes(args, dfs)
        else:
            try:
                dfs = data.import_dataframes(args, df_list=["groundtruth",
                                                            "eventinfo",
                                                            "features",
                                                            "prediction"])
            except FileNotFoundError:
                print("[!] Dataframes not found! Try processing first?")

        # Classification functions
        df_features = data.generate_feature_vectors(df_eventinfo)
        df_prediction = data.classify_feature_vectors(df_features)

        # Evaluation and export functions
        data.export_dataframes(args, {"df_eventinfo": df_eventinfo,
                                      "df_features": df_features,
                                      "df_prediction": df_prediction,
                                      "df_groundtruth": df_groundtruth})
        data.evaluate_results(args, df_groundtruth, df_prediction)
        data.plot_result(args, df_groundtruth, df_prediction,
                         key="EXT_CHM", flag="cumu_comparison")
        data.plot_result(args, dfs["groundtruth"], dfs["prediction"],
                         key="EXT_CHM", flag="false_positives")
        data.plot_result(args, dfs["groundtruth"], dfs["prediction"],
                         key="EXT_CHM", flag="false_negatives")


def set_parameters():
    """Dashboard for setting parameters for each processing stage of algorithm.

    Distinct from command line arguments. For this program, arguments are used
    for file I/O, directory selection, etc. These parameters affect the image
    processing and analysis parts of the algorithm instead."""

    params = {
        # Robust PCA/motion estimation
        "queue_size": 21,
        "ialm_lmbda": 0.01,
        "ialm_tol": 0.001,
        "ialm_maxiter": 100,
        "ialm_darker": True,

        # Thresholding
        "thr_type": 3,     # value of cv2.THRESH_TOZERO option
        "thr_value": 10,

        # Greyscale processing
        "grey_op_SE": [(2, 2), (3, 3)]
    }

    return params


if __name__ == "__main__":
    # Command line arguments used for specifying file I/O.
    # (NOT algorithm parameters. See set_parameters() for parameter choices.)
    parser = ap.ArgumentParser()

    # General arguments for video file I/O (should not change for testing)
    parser.add_argument("-d",
                        "--video_dir",
                        help="Path to directory containing video file",
                        default="videos/"
                        )
    parser.add_argument("-f",
                        "--filename",
                        help="Name of video file",
                        default="NPD 541 CHSW 2019 June 14.mp4"
                        )
    parser.add_argument("-t",
                        "--timestamp",
                        help="Specified starting timestamp for video",
                        default="2019-06-14 00:00:00.000000000"
                        )
    # Ground truth "groundtruth.csv" only valid for ch04_20170518205849.mp4
    parser.add_argument("-g",
                        "--groundtruth",
                        help="Path to ground truth file associated with video",
                        default="groundtruth/groundtruth.csv"
                        )

    # Flags to determine which program functionality should be run in testing
    parser.add_argument("-e",
                        "--extract",
                        help="Extract frames to HH:MM subfolders",
                        action="store_true",
                        default=False
                        )
    parser.add_argument("-p",
                        "--process",
                        help="Load and process frames from HH:MM subfolders",
                        action="store_true",
                        default=True
                        )
    parser.add_argument("-a",
                        "--analyse",
                        help="Analyse results by comparing to ground truth",
                        action="store_true",
                        default=True
                        )

    # Arguments for running image processing/analysis tests
    parser.add_argument("-l",
                        "--load",
                        help="Specify indices to load previously saved frames",
                        nargs=2,
                        type=int,
                        metavar=('START_INDEX', 'END_INDEX'),
                        default=([55000, 55100])
                        )
    parser.add_argument("-c",
                        "--custom_dir",
                        help="Custom directory for saving various things",
                        default="tests/load-frame/"
                        )
    parser.add_argument("-v",
                        "--visual",
                        help="Output visualization of frame processing",
                        default=True
                        )
    parser.add_argument("-n",
                        "--chimney",
                        help="Bottom corners which define chimney edge",
                        default=[(798, 449), (1150, 435)]
                        )
    arguments = parser.parse_args()

    # Repeatedly used default directory to ensure standardization. Storing here
    # because it is a derived from only arguments.
    arguments.default_dir = (arguments.video_dir +
                             os.path.splitext(arguments.filename)[0] + "/")

    parameters = set_parameters()
    main(arguments, parameters)
