import swiftwatcher.video_processing as vid
import swiftwatcher.data_analysis as data
import pandas as pd
import argparse as ap
import os
import time
import json
from pathlib import Path
from os import fspath


def load_config(video_dir, config_dir):
    if not config_dir.exists():
        config_dir.mkdir(parents=True, exist_ok=True)

    filepaths = [f for f in video_dir.iterdir() if f.is_file()]

    config_list = []
    for filepath in filepaths:
        config_filepath = config_dir/(filepath.stem + ".json")
        if not config_filepath.exists():
            config = {
                "name": filepath.name,
                "timestamp": "00:00:00.000000",
                "corners": vid.select_corners(),
            }
            with config_filepath.open(mode="w") as write_file:
                json.dump(config, write_file, indent=4)
        else:
            with config_filepath.open(mode="r") as read_file:
                config = json.load(read_file)

        config["src_filepath"] = filepath
        config["base_dir"] = filepath.parent / filepath.stem
        config_list.append(config)

    return config_list


def main():
    """To understand the current configuration of the algorithm, please look
    to the following functions, which are outside of main() below:

    - args: command-line arguments, used for file I/O, set by
        if __name__ == "__main__": block of code.
    - params: algorithm parameters, used to tweak processing stages, set by
        set_parameters() function."""

    configs = load_config(video_dir=Path.cwd()/"videos",
                          config_dir=Path.cwd()/"videos"/"configs")
    for config in configs:
        events = vid.swift_counting_algorithm(config)

        if len(events) > 0:
            features = data.generate_feature_vectors(events)
            labels = data.generate_classifications(features)
            data.export_results(config, labels)

        else:
            print("No detected chimney swifts in specified video.")


if __name__ == "__main__":
    main()
