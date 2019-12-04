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

import swiftwatcher_refactor.image_processing.data_structures as ds
import swiftwatcher_refactor.io.video_io as vio
import swiftwatcher_refactor.data_analysis.segment_tracking as st
import swiftwatcher_refactor.io.ui as ui



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

    ui.start_status(frame_path.parent.name)

    reader = vio.FrameReader(frame_path, start, end)
    queue = ds.FrameQueue()
    tracker = st.SegmentTracker(roi_mask)

    while queue.frames_processed < reader.total_frames:
        frames, frame_numbers = reader.get_n_frames(n=queue.maxlen)
        queue.fill_queue(frames, frame_numbers)
        queue.preprocess_queue(crop_region, resize_dim)
        queue.segment_queue()

        while not queue.is_empty():
            tracker.set_current_frame(queue.pop_frame())
            cost_matrix = tracker.formulate_cost_matrix()
            assignments = tracker.apply_hungarian_algorithm(cost_matrix)
            tracker.interpret_assignments(assignments)
            tracker.link_matching_segments()
            tracker.check_for_events()
            tracker.cache_current_frame()

        ui.frames_processed_status(queue.frames_processed, reader.total_frames)

    return tracker.detected_events
