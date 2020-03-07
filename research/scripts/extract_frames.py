import swiftwatcher.io_video as vio
import swiftwatcher.ui as ui
import sys

import argparse
from pathlib import Path

import datetime
import cv2


def main(filepaths):
    for filepath in filepaths:
            vio.validate_filepaths(filepath)
            vio.validate_video_files(filepath)
            extract_video_frames(filepath,
                                 filepath.parent/filepath.stem/"frames",
                                 ["%hour%", "h", "%minute%", "m"],
                                 ["%video_filename%", "_",
                                  "%frame_number%", "_",
                                  "%hour%", "h", "%minute%", "m",
                                  "%second%", "s", "%microsecond%", "us"])


def extract_video_frames(video_filepath,
                         base_dir,
                         subdir_name_scheme,
                         file_name_scheme):
    """Extract frames from video file and save them to image files
    according to filename and dirname schemes.

    Schemes are passed as list of either string literals or variables
    from a limited list of values:
        -video_filename
        -frame_number
        -datetime class attributes
            -year
            -month
            -day
            -hour
            -minute
            -second
            -millisecond
            -microsecond"""

    # Create initial datetime object
    start_time = datetime.datetime(100, 1, 1, 0, 0, 0, 0)

    # Ensure that video filepath is a Path object
    base_dir = Path(base_dir)
    video_filepath = Path(video_filepath)
    video_filename = video_filepath.name

    stream = cv2.VideoCapture(str(video_filepath))
    while True:
        # Determine attributes associated with frame
        frame_number = int(stream.get(cv2.CAP_PROP_POS_FRAMES))
        frame_ms = int(stream.get(cv2.CAP_PROP_POS_MSEC))
        timestamp = start_time + datetime.timedelta(milliseconds=frame_ms)

        # Attempt to load new frame
        success, frame = stream.read()
        if not success:
            break

        # Create sub directory string
        sub_dir = ""
        for item in subdir_name_scheme:
            if item[0] == "%" and item[-1] == "%":
                if item in ["%hour%", "%minute%", "%second%"]:
                    num = eval("timestamp." + item.replace("%", ""))
                    sub_dir += ("{:02d}".format(num))
                elif item in ["%microsecond%"]:
                    num = eval("timestamp." + item.replace("%", ""))
                    sub_dir += ("{:03d}".format(num))
                elif item in ["%video_filename%", "%frame_number%"]:
                    sub_dir += str((eval(item.replace("%", ""))))
                else:
                    sys.stderr.write("Error: {} not valid for naming scheme."
                                     .format(item))
            else:
                sub_dir += item

        # Create file name string
        file_name = ""
        for item in file_name_scheme:
            if item[0] == "%" and item[-1] == "%":
                if item in ["%hour%", "%minute%", "%second%"]:
                    num = eval("timestamp." + item.replace("%", ""))
                    file_name += "{:02d}".format(num)
                elif item in ["%microsecond%"]:
                    num = eval("timestamp." + item.replace("%", ""))
                    file_name += "{:06d}".format(num)
                elif item in ["%video_filename%", "%frame_number%"]:
                    file_name += str((eval(item.replace("%", ""))))
                else:
                    sys.stderr.write("Error: {} not valid for naming scheme."
                                     .format(item))
            else:
                file_name += item

        # Create output_directory if it doesn't exist
        output_dir = base_dir / sub_dir
        if not output_dir.exists():
            Path.mkdir(output_dir, parents=True)

        # Write frame to output_directory
        filepath = output_dir / (file_name + ".png")
        if not filepath.exists():
            cv2.imwrite(str(filepath), frame)


def parse_filepaths():
    """Parse all command line arguments as filepaths."""

    parser = argparse.ArgumentParser()
    parser.add_argument("filepaths", nargs="*")
    args = parser.parse_args()

    args.filepaths = [Path(filepath) for filepath in args.filepaths]

    return args.filepaths


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = parse_filepaths()
    else:
        paths = ui.select_filepaths()
        vio.validate_filepaths(paths)

    main(paths)
