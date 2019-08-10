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
    results_list = []
    config_list = []
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
                dfs = data.import_dataframes(config["test_dir"],
                                             ["groundtruth"])
                dfs["eventinfo"] = df_eventinfo
            else:
                dfs = data.import_dataframes(config["test_dir"],
                                             ["groundtruth", "eventinfo"])

            dfs["features"] = data.generate_feature_vectors(dfs["eventinfo"])
            dfs["prediction"] = data.generate_classifications(dfs["features"])
            dfs["comparison_before"], dfs["comparison"] \
                = data.generate_comparison(config,
                                           dfs["prediction"],
                                           dfs["groundtruth"])
            results = data.evaluate_results(config["test_dir"],
                                            dfs["comparison"])
            dfs.update(results)
            data.export_dataframes(config["test_dir"], dfs)
            data.plot_result(config["test_dir"],  dfs["prediction"],
                             dfs["groundtruth"], flag="cumu_comparison")
            results_list.append(results)
            config_list.append(config)

    data.feature_engineering(args, config_list, results_list)

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
    main()
