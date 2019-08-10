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

    video_dir = "videos/"
    args.custom_dir = ""
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

    videos = video_dictionary

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


if __name__ == "__main__":
    main()
