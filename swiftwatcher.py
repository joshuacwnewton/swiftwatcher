import swiftwatcher.process_video as pv
import os
import argparse as ap
import numpy as np
import cv2


def main(args):
    # Code to extract all frames from video and save them to files
    if args.extract:
        pv.extract_frames(args.video_dir, args.filename)

    # Code to load previously extracted frames and apply image processing to them
    if args.reuse:
        # Initialize parameters necessary to load previously extracted frames
        load_directory = (args.video_dir +  # Assumed convention from pv.extract_frames()
                          os.path.splitext(args.filename)[0])
        load_index = args.reuse[0]          # Index of frame to load next
        total_frames = args.reuse[1]        # Total number of frames to load and process

        # Create frameStack object
        stack_size = 20
        frame_stack = pv.FrameStack(args.video_dir, args.filename, stack_size=stack_size)
        frame_stack.stream.release()  # VideoCapture not needed if frames are being reused

        counter = 0
        while frame_stack.frames_read < total_frames:
            # Load frame with specified index
            success = frame_stack.load_frame_from_file(load_directory, load_index)

            if success:
                # Preprocessing
                frame_stack.convert_grayscale()
                frame_stack.crop_frame(corners=[(745, 617), (920, 692)])  # top-left [w,h], bottom-right [w,h]

                # Robust PCA background subtraction
                s = 10
                resized = cv2.resize(frame_stack.stack[0],
                                     (round(frame_stack.stack[0].shape[1]*s), round(frame_stack.stack[0].shape[0]*s)),
                                     interpolation=cv2.INTER_AREA)
                cv2.imwrite("{0}/RPCA/{1}a_original.jpg".format(load_directory, counter), resized)
                counter += 1
                frame_stack.frame_to_column()
                input_data = frame_stack.concatenate_stack()
                if frame_stack.frames_read >= stack_size:
                    low_rank, sparse = frame_stack.rpca_decomposition(input_data)
                    for i in range(stack_size):
                        reshaped_s = (np.reshape(sparse[:, i], (75, 175)))
                        abs_s = np.absolute(reshaped_s)
                        # inted_s = abs_s.astype(int)
                        resized_s = cv2.resize(abs_s,
                                              (round(abs_s.shape[1] * s), round(abs_s.shape[0] * s)),
                                              interpolation=cv2.INTER_AREA)

                        reshaped_lr = (np.reshape(low_rank[:, i], (75, 175)))
                        # inted_lr = reshaped_lr.astype(int)
                        resized_lr = cv2.resize(reshaped_lr,
                                                (round(reshaped_lr.shape[1] * s), round(reshaped_lr.shape[0] * s)),
                                                interpolation=cv2.INTER_AREA)

                        i = (stack_size-1) - i
                        cv2.imwrite("{0}/RPCA/{1}b_sparse.jpg".format(load_directory, i), resized_s)
                        cv2.imwrite("{0}/RPCA/{1}c_low_rank.jpg".format(load_directory, i), resized_lr)
                    test=None

                # Save files a custom folder within the load directory
                # frame_stack.save_frame_to_file(load_directory, folder_name=args.custom_dir)
                load_index += (1 + frame_stack.delay)

    # Code to process frames as they are being read
    else:
        # TODO: Real-time analysis code here
        todo = None


if __name__ == "__main__":
    # This file uses default configs which refer to the current best dataset available:
    # videos/ch04_20170518205849.mp4 with no custom subfolder (i.e. HH:MM subfolders)

    # Example 1: python3 swiftwatcher.py -e
    #                                    -d "videos/additional_videos/gdrive_swiftvideos/"
    #                                    -f "ch02_20160513200533.mp4"
    # Extracts frames from non-default video file.

    # Example 2: python3 swiftwatcher.py -r 16200 1800
    #                                    -c "Cropped-files_00:09-00:10"
    # Processes 1800 frames (previously extracted from default video file) starting at
    # frame 16200 (00:09) and saves them in custom subfolder called "Cropped-files_00:09-00:10".

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
                        metavar=('START_FRAME', 'TOTAL_FRAMES'),
                        default=([16200, 1800])
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
