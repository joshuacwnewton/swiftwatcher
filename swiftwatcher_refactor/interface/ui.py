"""
    Contains user interface functionality, including GUI elements and
    CLI prompts.
"""

import sys
import argparse
from os import fspath
from pathlib import Path

import tkinter as tk
from tkinter import filedialog

import cv2
import json


###############################################################################
#                    FILE SELECTION FUNCTIONS BEGIN HERE                      #
###############################################################################


def select_filepaths():
    """Select files using tk gui and return their paths if valid."""

    filepaths, keep_selecting = [], True

    while keep_selecting:
        filepaths = gui_append_files(filepaths)
        keep_selecting = prompt_additional_selection(filepaths)

    return filepaths


def gui_append_files(existing_file_list):
    """Select files using tk gui and append new files to passed list."""

    # See: https://stackoverflow.com/questions/1406145/
    root = tk.Tk()
    root.withdraw()

    # Tk dialog to select additional files
    files = filedialog.askopenfilenames(parent=root,
                                        title='Choose the files '
                                              'you wish to '
                                              'analyse.')

    # Convert (unique) selected files into Path objects
    new_file_list = (existing_file_list +
                     ([Path(f) for f in list(root.tk.splitlist(files))
                      if Path(f) not in existing_file_list]))

    if not new_file_list:
        sys.stderr.write("[!] Error: No file selected.")
        sys.exit()

    return new_file_list


def prompt_additional_selection(file_list):
    """Display file list to console and prompt to select additional
    files."""

    # Print current list of selected files
    print("[*] Video files to be analysed: ")
    filenames = ["[-]     {}".format(f.name) for f in file_list]
    print(*filenames, sep="\n")

    # Prompt user for additional file selection
    ipt = input("[*] Are there additional files you would like to "
                "select? (Y/N) \n"
                "[-]     Input: ")

    if ipt.lower() is "y":
        return True
    else:
        return False


###############################################################################
#               CHIMNEY CORNER SELECTION FUNCTIONS BEGIN HERE                 #
###############################################################################


def select_chimney_corners(filepath):
    """Pops up a window showing the first frame of a video file and
    prompts the user to select two corners. Returns a list containing
    the two points. Exits Python if the user closes the window."""

    def click_and_update(event, x, y, flags, param):
        """Callback function to record mouse coordinates on click, and
        update instructions to user accordingly."""

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(corners) < 2:
                corners.append((int(x), int(y)))
                cv2.circle(image, corners[-1], 5, (0, 0, 255), -1)
                cv2.imshow("image", image)
                cv2.resizeWindow('image',
                                 int(0.5*image.shape[1]),
                                 int(0.5*image.shape[0]))

            if len(corners) == 1:
                cv2.setWindowTitle("image",
                                   "Click on corner 2")

            if len(corners) == 2:
                cv2.setWindowTitle("image",
                                   "Type 'y' to keep,"
                                   " or 'n' to select different corners.")

    # Create and show window displaying first frame of video, and attach
    # a custom callback function
    stream = cv2.VideoCapture(fspath(filepath))
    success, image = stream.read()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", click_and_update)
    cv2.setWindowTitle("image", "Click on corner 1")

    # Create a copy so frame can be reset once image has been drawn on
    clone = image.copy()

    corners = []
    while True:
        # Display image and wait for user input (click -> click_and_update())
        cv2.imshow("image", image)
        cv2.resizeWindow('image',
                         int(0.5 * image.shape[1]),
                         int(0.5 * image.shape[0]))
        cv2.waitKey(1)

        if len(corners) == 2:
            key = cv2.waitKey(2000) & 0xFF

            # Indicates selected corners are not good, so resets state
            if chr(key).lower() == "n":
                image = clone.copy()
                corners = []
                cv2.setWindowTitle("image",
                                   "Click on corner 1")

            # Indicates selected corners are acceptable
            elif chr(key).lower() == "y":
                break

        # Indicates window has been closed prematurely
        if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) == 0:
            sys.stderr.write("[!] Error: Window closed without selecting"
                             "any chimney corners.\n")
            sys.exit()

    # Corners have been selected and approved, so close window
    cv2.destroyAllWindows()

    return corners


def get_corners_from_file(parent_directory):
    """Non-GUI alternative to select_video_corners to save time when
    running experiments."""

    with open(str(parent_directory / "attributes.json")) as json_file:
        video_attributes = json.load(json_file)

        # Convert from string to individual integer values
        video_attributes["corners"] \
            = [(int(video_attributes["corners"][0][0]),
                int(video_attributes["corners"][0][1])),
               (int(video_attributes["corners"][1][0]),
                int(video_attributes["corners"][1][1]))]

    return video_attributes["corners"]


###############################################################################
#                 CLI ARGUMENT PARSING FUNCTIONS BEGIN HERE                   #
###############################################################################


def parse_filepaths():
    """Parse all command line arguments as filepaths."""

    parser = argparse.ArgumentParser()
    parser.add_argument("filepaths", nargs="*")
    args = parser.parse_args()

    args.filepaths = [Path(filepath).resolve() for filepath in args.filepaths]

    return args.filepaths


def parse_filepath_and_framerange():
    """Parse named arguments for filepath, starting frame, and ending
    frame.."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()

    return Path(args.filepath).resolve(), int(args.start), int(args.end)


def status_update(frames_processed, total_frames):
    """Provide frequent status updates on how many frames have been
    processed"""
    if frames_processed % 25 is 0 and frames_processed is not 0:
        sys.stdout.write("\r[-]     {0}/{1} frames processed."
                         .format(frames_processed, total_frames))
        sys.stdout.flush()


###############################################################################
#                    STATUS UPDATE FUNCTIONS BEGIN HERE                       #
###############################################################################


def start_status(video_name):
    sys.stdout.write("[*] Now processing {}.\n".format(video_name))


def frames_processed_status(frames_processed, total_frames):
    sys.stdout.write("\r[-]     {0}/{1} frames processed.".format(
        frames_processed, total_frames))
    sys.stdout.flush()

