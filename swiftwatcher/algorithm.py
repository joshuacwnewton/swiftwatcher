"""
    algorithm.py contains functions which take individual frame
    processing stages in video_processing.py and combine them to create
    coherent algorithms.
"""

import swiftwatcher.video_processing as vid
import pandas as pd
import cv2
import sys
import glob


def create_dataframe(passed_list):
    """Convert list of events to pandas dataframe."""
    if passed_list:
        dataframe = pd.DataFrame(passed_list,
                                 columns=list(passed_list[0].keys())
                                 ).astype('object')
        dataframe["TMSTAMP"] = pd.to_datetime(dataframe["TMSTAMP"])
        dataframe["TMSTAMP"] = dataframe["TMSTAMP"].dt.round('us')
        dataframe.set_index(["TMSTAMP", "FRM_NUM"], inplace=True)
    else:
        dataframe = pd.DataFrame(columns=[])

    return dataframe


def swift_counting_algorithm(config):
    """Full algorithm which uses FrameQueue methods to process an entire video
    from start to finish."""

    print("[*] Now processing {}.".format(config["name"]))
    # print("[-]     Status updates will be given every 100 frames.")

    fq = vid.FrameQueue(config)
    while fq.frames_processed < fq.src_framecount:
        success = False

        # Store state variables in case video processing glitch occurs
        # (e.g. due to poorly encoded video)
        pos = fq.stream.get(cv2.CAP_PROP_POS_FRAMES)
        read = fq.frames_read
        proc = fq.frames_processed

        try:
            success, frame = fq.stream.read()
            frame_number = int(fq.stream.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = fq.fn_to_ts(fq.stream.get(cv2.CAP_PROP_POS_FRAMES))

            # Load frames until queue is filled
            if fq.frames_read < (fq.queue_size - 1):
                fq.load_frame(frame, frame_number, timestamp)
                fq.preprocess_frame()
                # fq.segment_frame() (not needed until queue is filled)
                # fq.match_segments() (not needed until queue is filled)
                # fq.analyse_matches() (not needed until queue is filled)

            # Process queue full of frames
            elif (fq.queue_size - 1) <= fq.frames_read < fq.src_framecount:
                fq.load_frame(frame, frame_number, timestamp)
                fq.preprocess_frame()
                fq.segment_frame()
                fq.match_segments()
                fq.analyse_matches()

            # Load blank frames until queue is empty
            elif fq.frames_read == fq.src_framecount:
                fq.load_frame(None, None, None)
                # fq.preprocess_frame() (not needed for blank frame)
                fq.segment_frame()
                fq.match_segments()
                fq.analyse_matches()

        except Exception as e:
            # TODO: Print statements here should be replaced with logging
            # Previous: print("[!] Error has occurred, see: '{}'.".format(e))
            # I'm not satisfied with how unexpected errors are handled
            fq.failcount += 1

            # Increment state variables to ensure algorithm doesn't get stuck
            if fq.stream.get(cv2.CAP_PROP_POS_FRAMES) == pos:
                fq.stream.grab()
            if fq.frames_read == read:
                fq.frames_read += 1
            if fq.frames_processed == proc:
                fq.frames_processed += 1

        if success:
            fq.failcount = 0
        else:
            fq.failcount += 1

        # Break if too many sequential errors
        if fq.failcount >= 10:
            # TODO: Print statements here should be replaced with logging
            # Previous: print("[!] Too many sequential errors have occurred. "
            #                 "Halting algorithm...")
            # I'm not satisfied with how unexpected errors are handled
            fq.frames_processed = fq.src_framecount + 1

        # Status updates
        if fq.frames_processed % 25 is 0 and fq.frames_processed is not 0:
            sys.stdout.write("\r[-]     {0}/{1} frames processed.".format(
                fq.frames_processed, fq.src_framecount))
            sys.stdout.flush()

    if fq.event_list:
        df_eventinfo = create_dataframe(fq.event_list)
    else:
        df_eventinfo = []
    print("")

    return df_eventinfo


def swift_counting_algorithm_from_frames(config, start, end):
    """"""

    print("[*] Now processing {}.".format(config["name"]))

    # Initialize frame queue and starting values
    fq = vid.FrameQueue(config)
    frame_number = start

    total_frames = end - start + 1
    while fq.frames_processed < total_frames:
        # Load frame from file
        frame_list = glob.glob(str(config["base_dir"]/"frames"/"*"/
                                   ("*_" + str(frame_number) + "_*" + ".png")))
        frame = cv2.imread(frame_list[0])

        # Load frames until queue is filled
        if fq.frames_read < (fq.queue_size - 1):
            fq.load_frame(frame, frame_number, fq.fn_to_ts(frame_number))
            fq.preprocess_frame()
            # fq.segment_frame() (not needed until queue is filled)
            # fq.match_segments() (not needed until queue is filled)
            # fq.analyse_matches() (not needed until queue is filled)

        # Process queue full of frames
        elif (fq.queue_size - 1) <= fq.frames_read < total_frames:
            fq.load_frame(frame, frame_number, fq.fn_to_ts(frame_number))
            fq.preprocess_frame()
            fq.segment_frame()
            fq.match_segments()
            fq.analyse_matches()

        # Load blank frames until queue is empty
        elif fq.frames_read == total_frames:
            fq.load_frame(None, None, None)
            # fq.preprocess_frame() (not needed for blank frame)
            fq.segment_frame()
            fq.match_segments()
            fq.analyse_matches()

        if fq.frames_processed % 25 is 0 and fq.frames_processed is not 0:
            sys.stdout.write("\r[-]     {0}/{1} frames processed.".format(
                fq.frames_processed, total_frames))
            sys.stdout.flush()

        frame_number += 1

    df_eventinfo = create_dataframe(fq.event_list)

    return df_eventinfo
