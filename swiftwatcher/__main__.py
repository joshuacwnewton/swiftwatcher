# Algorithm components
import swiftwatcher.video_processing as vid
import swiftwatcher.data_analysis as data

# File I/O
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog


def load_config(video_dir):

    filepaths = [f for f in video_dir.iterdir() if f.is_file()]

    config_list = []
    for filepath in filepaths:
        config_dir = filepath.parent/filepath.stem
        if not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)

        config_filepath = config_dir/(filepath.stem + ".json")
        if not config_filepath.exists():
            config = {
                "name": filepath.name,
                "timestamp": "00:00:00.000000",
                "corners": vid.select_corners(filepath),
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

    root = tk.Tk()
    root.withdraw()
    dirpath = Path(filedialog.askdirectory(parent=root, initialdir="/",
                                           title='Please select a directory '
                                                 'containing the videos you '
                                                 'wish to analyse.'))

    configs = load_config(dirpath)
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
