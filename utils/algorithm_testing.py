from datetime import date
from glob import glob
from pathlib import Path

import swiftwatcher.video_processing as vid
import swiftwatcher.data_analysis as data
import cv2
import sys


def swift_counting_algorithm_from_files(config, start, end):
    """"""

    print("[*] Now processing {}.".format(config["name"]))

    # Initialize frame queue and starting values
    fq = vid.FrameQueue(config)
    frame_number = start
    total_frames = end - start + 1

    while fq.frames_processed < total_frames:
        # If frames left, read from file. If no frames left, flush out queue.
        if fq.frames_read < total_frames:
            frame_list = glob(str(config["base_dir"] / "frames" / "*" /
                                  ("*_" + str(frame_number) + "_*.png")))
            frame = cv2.imread(frame_list[0])
            fq.load_frame(frame, frame_number, fq.fn_to_ts(frame_number))
            fq.preprocess_frame()
        else:
            fq.load_frame(None, None, None)

        # If queue has enough frames for analysis, begin algorithm
        if fq.frames_read >= fq.queue_size:
            # Image processing stages
            fq.segment_frame()
            fq.match_segments()
            fq.analyse_matches()

            # Testing stages
            fq.export_segments()

        if fq.frames_processed % 25 is 0 and fq.frames_processed is not 0:
            sys.stdout.write("\r[-]     {0}/{1} frames processed.".format(
                fq.frames_processed, total_frames))
            sys.stdout.flush()

        frame_number += 1

    df_eventinfo = data.create_dataframe(fq.event_list)

    return df_eventinfo


def export_segments(fq):
    if fq.seg_properties[0]:
        test = None


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


def dataframe_to_csv(test_dir, input_dataframe):
    if not test_dir.exists():
        Path.mkdir(test_dir, parents=True)

    input_dataframe.to_csv(str(test_dir / "events.csv"))


def csv_to_dataframe():
    """"""

