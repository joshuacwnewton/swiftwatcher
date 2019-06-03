import swiftwatcher.process_video as pv
import os
import argparse as ap

# Used for performance evaluation, could be moved to process_video()
import numpy as np

# TODO: Switch from "extract frames -> reload frames" to real-time analysis
# Written this way to save time when testing with specific timestamps


def main(args, params):
    """Count swift behavior (entering/exiting chimney) from video frames.

    - args: command-line arguments, used for file I/O, set by
        if __name__ == "__main__": block of code.
    - params: algorithm parameters, used to tweak processing stages, set by
        set_parameters() function."""

    # Code to extract all frames from video and save them to image files
    if args.extract:
        pv.extract_frames(args.video_dir, args.filename)

    # Code to process previously extracted frames
    if args.load:
        # --------------------- INITIALIZATION STARTS ----------------------- #

        # File I/O
        load_directory = (args.video_dir + os.path.splitext(args.filename)[0])
        frame_to_load = args.load[0]
        num_frames_to_read = args.load[1] - args.load[0]

        # FrameQueue object (class for caching frames, processing frames)
        frame_queue = pv.FrameQueue(args.video_dir, args.filename,
                                    queue_size=params["queue_size"])
        frame_queue.stream.release()  # Videocapture not needed for frame reuse

        # Ground truth and measured stats
        ground_truth = np.genfromtxt('videos/groundtruth.csv',
                                     delimiter=',').astype(dtype=int)
        stats_array = np.array([]).reshape(0, 6)

        # ----- INITIALIZATION ENDS, FRAME PROCESSING ALGORITHM BEGINS ------ #

        while frame_queue.frames_read < num_frames_to_read:
            # Load frame into index 0 and apply preprocessing
            frame_queue.load_frame_from_file(load_directory, frame_to_load)
            frame_queue.convert_grayscale(algorithm="cv2 built-in")
            frame_queue.crop_frame(corners=params["corners"])
            frame_queue.frame_to_column()

            # Proceed only when enough frames are stored for motion estimation
            if frame_queue.frames_read > params["queue_center"]:
                # Segment frame using context from adjacent frames in queue
                processing_stages = \
                    frame_queue.segment_frame(index=params["queue_center"],
                                              visual=True)

                # Match bird segments from two sequential frames
                match_coords, match_stats, match_comparison = \
                    frame_queue.match_segments(visual=True)

                # Save results to image files for visual inspection
                if processing_stages is not None:
                    frame_queue.save_frame_to_file(load_directory,
                                                   frame=processing_stages,
                                                   index=params["queue_center"],
                                                   folder_name=args.custom_dir,
                                                   file_prefix="seg_",
                                                   scale=400)
                if match_comparison is not None:
                    frame_queue.save_frame_to_file(load_directory,
                                                   frame=match_comparison,
                                                   index=params["queue_center"],
                                                   folder_name=args.custom_dir,
                                                   scale=400)

                # Store relevant match stats in ground_truth data structure
                if 16200 <= (frame_to_load - params["queue_center"]) <= 16390:
                    total_birds = (match_stats["total_matches"] +
                                   match_stats["appeared_from_edge"] +
                                   match_stats["appeared_from_chimney"])
                    stats_list = [frame_to_load - params["queue_center"],
                                  total_birds,
                                  match_stats["appeared_from_chimney"],
                                  match_stats["appeared_from_edge"],
                                  match_stats["disappeared_to_chimney"],
                                  match_stats["disappeared_to_edge"]]
                    stats_array = np.vstack((stats_array,
                                             stats_list)).astype(int)
                    test = None

            frame_to_load += (1 + frame_queue.delay)
            if frame_queue.frames_read % 50 == 0:
                print("{0}/{1} frames processed."
                      .format(frame_queue.frames_read, num_frames_to_read))

        error = ground_truth - stats_array
        true_sums = np.sum(ground_truth[:, 1:6], axis=0)
        guessed_sums = np.sum(stats_array[:, 1:6], axis=0)
        ground_truth = np.hstack((ground_truth[:, 1:6], stats_array[:, 1:6]))
        test = None


def set_parameters():
    """Set parameters for each processing stage of algorithm.

    Distinct from command line arguments. For this program, arguments are saved
    for file I/O, directory selection, etc. These parameters, by comparison,
    affect the image processing and analysis aspects of the algorithm. """
    params = {
        # Relevant parameters for frame cropping
        "corners": [(760, 650), (921, 686)],

        # Relevant parameters for RPCA/motion estimation
        "queue_size": 21,
    }

    # Precalculating commonly used values
    params["queue_center"] = int((params["queue_size"] - 1) / 2)

    return params


if __name__ == "__main__":
    # Command line arguments used for specifying file I/O.
    # (NOT algorithm parameters. See set_parameters() for parameter choices.)
    parser = ap.ArgumentParser()
    parser.add_argument("-e",
                        "--extract",
                        help="Extract frames to HH:MM subfolders",
                        action="store_true"
                        )
    parser.add_argument("-l",
                        "--load",
                        help="Option to load previously saved frames",
                        nargs=2,
                        type=int,
                        metavar=('START_INDEX', 'END_INDEX'),
                        default=([16150, 16450])
                        )
    parser.add_argument("-d",
                        "--video_dir",
                        help="Path to directory containing video file",
                        default="videos/"
                        )
    parser.add_argument("-f",
                        "--filename",
                        help="Name of video file",
                        default="ch04_20170518205849.mp4"
                        )
    parser.add_argument("-c",
                        "--custom_dir",
                        help="Custom directory for extracted frame files",
                        default="Match Stats Test"
                        )
    arguments = parser.parse_args()

    parameters = set_parameters()
    main(arguments, parameters)
