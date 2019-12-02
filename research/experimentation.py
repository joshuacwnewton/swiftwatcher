"""
    Separate module containing functionality for experimentation
    (e.g. loading frames from files, visualizing frames at intermediate
    stages of the algorithm, and determining a directory to output
    test results to).
"""

import json
from pathlib import Path
from glob import glob
from datetime import date

from swiftwatcher_refactor.image_processing.data_structures import FrameQueue
from swiftwatcher_refactor.io.video_io import FrameReader


def generate_test_dir(parent_dir):
    """Generate test directory based on the following scheme:
        parent_dir/<today's date>/<ID of last test + 1>

    If no test has been run today, set ID to 1."""

    # Set base testing directory to today's date
    date_dir = parent_dir / str(date.today())

    if not date_dir.exists():
        # Date directory doesnt exist, so must be first test run today
        test_dir = date_dir / "1"

    else:
        # Fetch names of all subdirectories in date_dir, then get the max
        last_test_id = max([int(path.stem) for path in
            [Path(path_str) for path_str in glob(str(date_dir / "*/"))]])

        # Set test directory to last test incremented by one
        test_dir = date_dir / str(last_test_id + 1)

    return test_dir


def get_corners_from_file(parent_directory):
    """Non-GUI alternative to get_video_attributes for time saving."""

    with open(str(parent_directory / "attributes.json")) as json_file:
        video_attributes = json.load(json_file)

        # Convert from string to individual integer values
        video_attributes["corners"] \
            = [(int(video_attributes["corners"][0][0]),
                int(video_attributes["corners"][0][1])),
               (int(video_attributes["corners"][1][0]),
                int(video_attributes["corners"][1][1]))]

    return video_attributes["corners"]


def swift_counting_algorithm(frame_path, crop_region, resize_dim, roi_mask,
                             start, end):
    """A modified version of the swift_counting_algorithm() found in
    image_processing/primary_algorithm.py that allows the following
    additional functionality:

        -Reading specific frames from files
        -Saving visualizations at intermediate stages of the
        algorithm"""

    print("[*] Now processing {}.".format(frame_path.parent.stem))

    fq = FrameQueue()
    fr = FrameReader(frame_path, start, end)

    while fr.frames_read < fr.total_frames:
        frames, frame_numbers = fr.get_n_frames(n=fq.maxlen)
        fq.set_queue(frames, frame_numbers)
        fq.preprocess_queue(crop_region, resize_dim)
        fq.segment_queue()

    return fq
