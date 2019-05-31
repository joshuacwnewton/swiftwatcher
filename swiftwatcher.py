import swiftwatcher.process_video as pv
import os
import argparse as ap

# Used for performance evaluation, could be moved to process_video()
import numpy as np

# TODO: Switch from "extract frames -> reload frames" to real-time analysis
# Written this way to save time when testing with specific timestamps


def main(args):
    # Code to extract all frames from video and save them to image files
    if args.extract:
        pv.extract_frames(args.video_dir, args.filename)

    # Code to process previously extracted frames
    if args.load:
        # Initialize parameters necessary to load previously extracted frames
        frame_to_load = args.load[0]
        num_frames_to_read = args.load[1] - args.load[0]
        # Directory follows assumed convention from pv.extract_frames()
        load_directory = (args.video_dir +
                          os.path.splitext(args.filename)[0])

        # Initialize FrameQueue object
        queue_size = 21  # Determines number of frames kept in memory at once
        queue_center = int((queue_size - 1) / 2)
        frame_queue = pv.FrameQueue(args.video_dir, args.filename,
                                    queue_size=queue_size)
        frame_queue.stream.release()  # Videocapture not needed for frame reuse

        # Initialize data structures for bird counting stats
        bird_count = np.array([])
        frame_cc_prev = None
        ground_truth = np.genfromtxt('videos/groundtruth.csv',
                                     delimiter=',').astype(dtype=int)

        while frame_queue.frames_read < num_frames_to_read:
            # Load frame into index 0 and apply preprocessing
            frame_queue.load_frame_from_file(load_directory, frame_to_load)
            frame_queue.convert_grayscale(index=0)
            frame_queue.crop_frame(index=0, corners=args.crop)
            frame_queue.frame_to_column(index=0)

            # Proceed only when enough frames are stored for motion estimation
            if frame_queue.frames_read > queue_center:
                # Segment frame using context from adjacent frames in queue
                num_cc, frame_cc, processing_stages = \
                    frame_queue.segment_frame(index=queue_center,
                                              visual=True)

                # Match bird segments from two sequential frames
                match_coords, match_comparison = \
                    frame_queue.match_segments(frame=frame_cc,
                                               frame_prev=frame_cc_prev,
                                               visual=True)
                frame_cc_prev = frame_cc  # Store frame for future matching

                # Save results to image files for visual inspection
                if processing_stages is not None:
                    frame_queue.save_frame_to_file(load_directory,
                                                   frame=processing_stages,
                                                   index=queue_center,
                                                   folder_name=args.custom_dir,
                                                   file_suffix="segmentation",
                                                   scale=400)
                if match_comparison is not None:
                    frame_queue.save_frame_to_file(load_directory,
                                                   frame=match_comparison,
                                                   index=queue_center,
                                                   folder_name=args.custom_dir,
                                                   scale=400)

            frame_to_load += (1 + frame_queue.delay)
            if frame_queue.frames_read % 50 == 0:
                print("{0}/{1} frames processed."
                      .format(frame_queue.frames_read, num_frames_to_read))

        # TODO: Properly measure and save statistics about bird matching
        # Calculate performance metrics from bird matching
        ground_truth = np.c_[ground_truth,
                             bird_count.reshape(-1, 1).astype(np.int)]
        error = ground_truth[:, 1] - ground_truth[:, 6]
        error_less = sum(error[error > 0])
        error_more = -1* sum(error[error < 0])


if __name__ == "__main__":
    # Default values currently used for testing to save time
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
                        default="Segmentation Test Folder"
                        )
    parser.add_argument("-p",
                        "--crop",
                        help="Corner coordinates for cropping.",
                        default=[(760, 650), (921, 686)]
                        )
    arguments = parser.parse_args()

    main(arguments)
