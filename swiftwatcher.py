# =================================== Ideas ================================= #
# -TODO: RPCA speed increase (crop regions, queue size, non-RPCA BS)
#
# -TODO: Try hard threshold at 10-15
#
# -TODO: Modify how angle is calculated.
# -TODO: Try increasing angle range? (Guess and check optimal decision boundary)
# -TODO: Analyse all videos for shared sources of error
# -TODO: Matching/feature: Area of segment (Avg?)
# -TODO: Matching/feature: Diagonal length of segment (Avg?)
# -TODO: Matching/feature: Relative position of segment
# -TODO: Overhaul how matching works to make more sense (costs)
# =========================================================================== #

import swiftwatcher.video_processing as vid
import swiftwatcher.data_analysis as data
import pandas as pd
import argparse as ap
import os
import time
import json
from pathlib import Path


def main(args):
    """To understand the current configuration of the algorithm, please look
    to the following functions, which are outside of main() below:

    - args: command-line arguments, used for file I/O, set by
        if __name__ == "__main__": block of code.
    - params: algorithm parameters, used to tweak processing stages, set by
        set_parameters() function."""

    # Debugging/testing modes of functionality
    for config_path in args.configs:
        with open(config_path) as json_file:
            config = json.load(json_file)
        config["src_filepath"] = Path("videos", config["name"])
        config["base_dir"] = Path(config["src_filepath"].parent,
                                  config["src_filepath"].stem)

        if args._extract:
            vid.extract_frames(args)
            pass
        else:
            config["base_dir"] = config["base_dir"] / "debug"
            config["test_dir"] = config["base_dir"] / args.custom_dir

        if args._process:
            start = time.time()
            df_eventinfo = vid.process_frames(args, config)
            end = time.time()

            elapsed_time = pd.to_timedelta((end - start), 's')
            print("[-] Frame processing took {}.".format(elapsed_time))

        if args._analyse:
            if args._process:
                dfs = data.import_dataframes(config["base_dir"],
                                             ["groundtruth"])
                dfs["eventinfo"] = df_eventinfo
                dfs["features"] = data.generate_feature_vectors(dfs["eventinfo"])
                dfs["prediction"] = data.generate_classifications(dfs["features"])
                dfs["comparison"] = data.generate_comparison(config,
                                                             dfs["prediction"],
                                                             dfs["groundtruth"])
                data.export_dataframes(config["test_dir"], dfs)
            else:
                dfs = data.import_dataframes(config["test_dir"],
                                             df_list=[
                                                 "groundtruth",
                                                 "eventinfo",
                                                 "features",
                                                 "prediction",
                                                 "comparison"
                                              ])
                dfs["comparison_before"], dfs["comparison"] \
                    = data.generate_comparison(config,
                                               dfs["prediction"],
                                               dfs["groundtruth"])
                data.export_dataframes(config["test_dir"], dfs)

            results = data.evaluate_results(config["test_dir"],
                                            dfs["comparison"])
            data.export_dataframes(config["test_dir"], results)
            data.plot_result(config["test_dir"],  dfs["prediction"],
                             dfs["groundtruth"], flag="cumu_comparison")
            data.plot_result(config["test_dir"], dfs["prediction"],
                             dfs["groundtruth"], flag="false_positives")
            data.plot_result(config["test_dir"], dfs["prediction"],
                             dfs["groundtruth"], flag="false_negatives")

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
                vid.full_algorithm(args, value)

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
                            default=False
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

    def set_processing_args():
        """Set arguments which relate to testing the algorithm, but that are
        unrelated to the functionality of the algorithm itself."""

        parser.add_argument("-z",
                            "--configs",
                            help="Config files for tests to be run",
                            default=[
                                "videos/configs/ch04_partial.json",
                                "videos/configs/june13_full-video.json",
                                "videos/configs/june14_full-video.json"
                            ]
                            )
        parser.add_argument("-c",
                            "--custom_dir",
                            help="Custom directory for saving various things",
                            default="tests/2019-08-06_full-video/"
                            )
        parser.add_argument("-v",
                            "--visual",
                            help="Output visualization of frame processing",
                            default=True
                            )

    parser = ap.ArgumentParser()
    set_program_flags()
    set_processing_args()
    arguments = parser.parse_args()
    main(arguments)
