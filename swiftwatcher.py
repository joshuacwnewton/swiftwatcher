# =================================== TODOs ================================= #
# execution-time:
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
# timestamp-precision:
# -TODO: Update empty-gt-generator for us timestamp precision.
# -TODO: Explore source of timestamp rounding error (off-by-one microsecond?)
#
# feature-engineering:
# -TODO: Implement/test idea for relative position feature.
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
            dfs["comparison"] = data.generate_comparison(dfs["eventinfo"],
                                                         dfs["groundtruth"])
            dfs["features"] = data.generate_feature_vectors(dfs["eventinfo"])
            dfs["prediction"] = data.generate_classifications(dfs["features"])
            data.export_dataframes(args, dfs)
        else:
            try:
                dfs = data.import_dataframes(args, df_list=["groundtruth",
                                                            "eventinfo",
                                                            "comparison",
                                                            "features",
                                                            "prediction"])
            except FileNotFoundError:
                print("[!] Dataframes not found! Try processing first?")

        data.evaluate_results(args, dfs["groundtruth"], dfs["prediction"])
        data.plot_result(args, dfs["groundtruth"], dfs["prediction"],
                         key="EXT_CHM", flag="cumu_comparison")
        data.plot_result(args, dfs["groundtruth"], dfs["prediction"],
                         key="EXT_CHM", flag="false_positives")
        data.plot_result(args, dfs["groundtruth"], dfs["prediction"],
                         key="EXT_CHM", flag="false_negatives")

        # Experimental function for testing new features/classifiers
        data.feature_engineering(args, dfs["comparison"])


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
    def set_program_flags():
        """Set flags which determine which modes of functionality the program
        should run in."""
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

    def set_fileio_args():
        """Set arguments relating to video file I/O."""
        parser.add_argument("-d",
                            "--video_dir",
                            help="Path to directory containing video file",
                            default="videos/"
                            )
        parser.add_argument("-f",
                            "--filename",
                            help="Name of video file",
                            default="NPD 460 CHSW 2019 June 13.mp4"
                            )
        parser.add_argument("-t",
                            "--timestamp",
                            help="Specified starting timestamp for video",
                            default="2019-06-13 00:00:00.000000000"
                            )

    def set_processing_args():
        """Set arguments which relate to testing the algorithm, but that are
        unrelated to the functionality of the algorithm itself."""

        parser.add_argument("-l",
                            "--load",
                            help="Specify indices to load frames",
                            nargs=2,
                            type=int,
                            metavar=('START_INDEX', 'END_INDEX'),
                            default=([0, -1])
                            )
        parser.add_argument("-c",
                            "--custom_dir",
                            help="Custom directory for saving various things",
                            default="tests/2019-07-23_full-video/"
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

    parser = ap.ArgumentParser()
    set_program_flags()
    set_fileio_args()
    set_processing_args()
    arguments = parser.parse_args()

    parameters = set_parameters()

    # Repeatedly used default directory to ensure standardization. Storing here
    # because it is a derived from only arguments.
    arguments.default_dir = (arguments.video_dir +
                             os.path.splitext(arguments.filename)[0] + "/")

    main(arguments, parameters)
