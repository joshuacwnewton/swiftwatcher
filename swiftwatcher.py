# =================================== Ideas ================================= #
# -TODO: RPCA speed increase (crop regions, queue size, non-RPCA BS)
#
# -TODO: Get both ch04 and 13/14 to use same GT processing
# -TODO: Automated batch processing (ch04, june 13, june 14 in one go)
#
# -TODO: Resize all cropped frames into same dimensions
# -TODO: Find alternative method to (3, 3) opening for removing chimney sway
#
# -TODO: Analyse all videos for shared sources of error
# -TODO: Matching/feature: Area of segment (Avg?)
# -TODO: Matching/feature: Diagonal length of segment (Avg?)
# -TODO: Matching/feature: Relative position of segment
# =========================================================================== #

import swiftwatcher.video_processing as vid
import swiftwatcher.data_analysis as data
import pandas as pd
import argparse as ap
import os
import time
import json


def main(args, params):
    """To understand the current configuration of the algorithm, please look
    to the following functions, which are outside of main() below:

    - args: command-line arguments, used for file I/O, set by
        if __name__ == "__main__": block of code.
    - params: algorithm parameters, used to tweak processing stages, set by
        set_parameters() function."""

    # Debugging/testing modes of functionality
    if args._extract:
        vid.extract_frames(args)
        pass
    if args._process:
        start = time.time()
        df_eventinfo = vid.process_frames(args, params)
        end = time.time()

        elapsed_time = pd.to_timedelta((end - start), 's')
        print("[-] Frame processing took {}.".format(elapsed_time))

        data.save_test_config(args, params)
    if args._analyse:
        if args._process:
            dfs = data.import_dataframes(args, ["groundtruth"])
            dfs["eventinfo"] = df_eventinfo
            dfs["features"] = data.generate_feature_vectors(dfs["eventinfo"])
            dfs["prediction"] = data.generate_classifications(dfs["features"])
            dfs["comparison"] = data.generate_comparison(dfs["prediction"],
                                                         dfs["groundtruth"])
            data.export_dataframes(args, dfs)
        else:
            try:
                dfs = data.import_dataframes(args, df_list=["groundtruth",
                                                            "eventinfo",
                                                            "features",
                                                            "prediction",
                                                            "comparison"])
            except FileNotFoundError:
                print("[!] Dataframes not found! Try processing first?")

        results = data.evaluate_results(args, dfs["comparison"])
        data.export_dataframes(args, results)
        data.plot_result(args,  dfs["prediction"],
                         dfs["groundtruth"], flag="cumu_comparison")
        data.plot_result(args, dfs["prediction"],
                         dfs["groundtruth"], flag="false_positives")
        data.plot_result(args, dfs["prediction"],
                         dfs["groundtruth"], flag="false_negatives")

        # Experimental function for testing new features/classifiers
        data.feature_engineering(args, results)

    # The set of steps which would be run by an end-user
    if args._production:
        args.video_dir = "videos/"
        args.custom_dir = ""
        videos = configuration(args.video_dir)

        for key, value in videos.items():
            args.filename = key
            args.default_dir = (args.video_dir +
                                os.path.splitext(args.filename)[0] + "/")
            videos[key]["eventinfo"] = \
                vid.full_algorithm(args, params, value)

            if not videos[key]["eventinfo"].empty:
                videos[key]["features"] = \
                    data.generate_feature_vectors(videos[key]["eventinfo"])
                videos[key]["prediction"] = \
                    data.generate_classifications(videos[key]["features"])
                data.plot_result(args, "EXT_CHM", videos[key]["prediction"])


def configuration(video_dir):
    files = [f for f in os.listdir(video_dir)
             if os.path.isfile(os.path.join(video_dir, f))]

    videos_json = []
    if 'config.json' in files:
        with open(video_dir + "config.json") as json_file:
            videos_json = json.load(json_file)

    video_dictionary = {}
    for filename in files:
        if filename in videos_json:
            video_dictionary[filename] = videos_json[filename]
        else:
            # Fetch corners from GUI
            pass

    with open(video_dir + "config.json", "w") as write_file:
        json.dump(video_dictionary, write_file, indent=4)

    return video_dictionary


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
                            "--_extract",
                            help="Extract frames to HH:MM subfolders",
                            action="store_true",
                            default=False
                            )
        parser.add_argument("-p",
                            "--_process",
                            help="Load and process frames from HH:MM folders",
                            action="store_true",
                            default=True
                            )
        parser.add_argument("-a",
                            "--_analyse",
                            help="Analyse results by comparing to groundtruth",
                            action="store_true",
                            default=True
                            )
        parser.add_argument("-b",
                            "--_production",
                            help="Batch process all videos in video directory",
                            action="store_true",
                            default=False
                            )

    def set_file_args():
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
                            # ch04_20170518205849.mp4
                            # NPD 460 CHSW 2019 June 13.mp4
                            # NPD 541 CHSW 2019 June 14.mp4
                            )
        parser.add_argument("-t",
                            "--timestamp",
                            help="Specified starting timestamp for video",
                            default="2019-06-13 00:00:00.000000"
                            )
        parser.add_argument("-n",
                            "--chimney",
                            help="Bottom corners which define chimney edge",
                            default=[(810, 435), (1150, 435)]
                            # [(748, 691), (920, 683)]  <- ch04_20170518
                            # [(810, 435), (1150, 435)] <- june 13
                            # [(798, 449), (1164, 423)] <- june 14
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
                            default=([6400, 9400])
                            )
        parser.add_argument("-c",
                            "--custom_dir",
                            help="Custom directory for saving various things",
                            default="/tests/2019-08-01_partial-fp-nofix/"
                            )
        parser.add_argument("-v",
                            "--visual",
                            help="Output visualization of frame processing",
                            default=True
                            )

    parser = ap.ArgumentParser()
    set_program_flags()
    set_file_args()
    set_processing_args()
    arguments = parser.parse_args()

    parameters = set_parameters()

    # Repeatedly used default directory to ensure standardization. Storing here
    # because it is a derived from only arguments.
    arguments.default_dir = (arguments.video_dir +
                             os.path.splitext(arguments.filename)[0] +
                             "/debug/")

    main(arguments, parameters)
