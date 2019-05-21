import swiftwatcher.process_video as pv
import os
import argparse as ap


def main(args):
    # Code to extract all frames from video and save them to files
    if args.extract:
        pv.extract_frames(args.video_dir, args.filename)

    # Code to load previously extracted frames and apply image processing to them
    if args.reuse:
        # Initialize parameters necessary to load previously extracted frames
        load_directory = args.video_dir + os.path.splitext(args.filename)[0]
        load_index = args.reuse[0]    # Index of frame to load next
        total_frames = args.reuse[1]  # Total number of frames to load and process

        # Create frameStack object
        frame_stack = pv.FrameStack(args.video_dir, args.filename)
        frame_stack.stream.release()  # VideoCapture not needed if frames are being reused

        while frame_stack.frames_read < total_frames:
            # Load frame with specified index
            success = frame_stack.load_frame_from_file(load_directory, load_index)

            if success:
                # Process frame
                frame_stack.convert_grayscale()
                frame_stack.crop_frame(corners=[(745, 617), (920, 692)])  # top-left [w,h], bottom-right [w,h]
                frame_stack.segment_frame()
                frame_stack.resize_frame(1000)

                # Save files a custom folder within the load directory
                frame_stack.save_frame_to_file(load_directory, folder_name=args.custom_dir)
                load_index += (1 + frame_stack.delay)
    else:
        # TODO: Real-time analysis code here
        todo = None


if __name__ == "__main__":
    # To run: python3 swiftwatcher.py --extract
    #                                 --reuse [START_FRAME TOTAL_FRAME]
    #                                 --video_dir <path to video folder>
    #                                 --filename <name of video file>
    #                                 --custom_dir <custom frame subfolder>

    # This file uses default configs which refer to the current best dataset available:
    # videos/ch04_20170518205849.mp4 with no custom subfolder (i.e. HH:MM subfolders)

    # Example: python swiftwatcher.py -r 16200 1800 -c "Cropped-files_00:09-00:10"
    # Processes 1800 frames (previously extracted from default video file) starting at
    # frame 16200 (00:09) and saves them in custom subfolder called "Cropped-files_00:09-00:10"

    parser = ap.ArgumentParser()
    parser.add_argument("-e",
                        "--extract",
                        help="Extract frames to HH:MM subfolders",
                        action="store_true"
                        )
    parser.add_argument("-r",
                        "--reuse",
                        help="Option to reuse previously saved frames",
                        nargs=2,
                        type=int,
                        metavar=('START_FRAME', 'TOTAL_FRAMES')
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
                        help="Custom directory for extracted frame image files",
                        default=None
                        )
    arguments = parser.parse_args()

    main(arguments)
