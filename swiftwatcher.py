import swiftwatcher.process_video as pv
import os
import argparse as ap
import cv2
import numpy as np

import utils.cm as cm
from scipy import ndimage as img


def main(args):
    # Code to extract all frames from video and save them to files
    if args.extract:
        pv.extract_frames(args.video_dir, args.filename)

    # Code to process previously extracted frames
    if args.reuse:
        # Initialize parameters necessary to load previously extracted frames
        # Directory follows assumed convention from pv.extract_frames()
        load_directory = (args.video_dir +
                          os.path.splitext(args.filename)[0])
        load_index = args.reuse[0]
        total_frames = args.reuse[1]

        # Create FrameQueue object
        queue_size = 20
        queue_center = int((queue_size-1)/2)
        frame_queue = pv.FrameQueue(args.video_dir, args.filename,
                                    queue_size=queue_size)
        frame_queue.stream.release()  # Not needed for frame reuse

        # Initialize data structures for bird counting stats
        bird_count = np.array([])
        ground_truth = np.genfromtxt('videos/groundtruth.csv',
                                     delimiter=',').astype(dtype=int)

        while frame_queue.frames_read < total_frames:
            # Load frame with specified index into FrameQueue object
            success = frame_queue.load_frame_from_file(load_directory,
                                                       load_index)

            # Process frame (grayscale, segmentation, etc.)
            if success:
                # Processing steps prior to motion estimation
                frame_queue.convert_grayscale()
                frame_queue.crop_frame(corners=args.crop)
                frame_queue.frame_to_column()

                if frame_queue.frames_read > queue_center:
                    # Choosing index such that RPCA will use adjacent frames
                    # (forward and backwards) to "queue_center" frame
                    lowrank, sparse = \
                        frame_queue.rpca_decomposition(index=queue_center,
                                                       darker_only=True)

                    # Apply bilateral filter to remove artifacts, retain birds
                    sparse_filtered = sparse
                    for i in range(2):
                        sparse_filtered = cv2.bilateralFilter(sparse_filtered,
                                                              d=7,
                                                              sigmaColor=15,
                                                              sigmaSpace=1)

                    # Retain strongest areas and discard the rest
                    _, sparse_thr = cv2.threshold(sparse_filtered,
                                                  thresh=35,
                                                  maxval=255,
                                                  type=cv2.THRESH_TOZERO)

                    # Segment using connected component labeling
                    retval, sparse_cc = \
                        cv2.connectedComponents(sparse_thr, connectivity=4)
                    # Scale CC image for visual clarity
                    if retval > 0:
                        sparse_cc = sparse_cc*(255/retval)

                    # Exclude background, count only foreground segments
                    num_components = retval - 1
                    if 16199 < (load_index - queue_center) < 16391:
                        bird_count = np.append(bird_count, num_components)

                    # Display different stages of processing in single image
                    separator = 255 * np.ones(shape=(1, frame_queue.width),
                                              dtype=np.uint8)
                    frame = np.reshape(frame_queue.queue[queue_center],
                                       (frame_queue.height, frame_queue.width))
                    sparse_frames = np.vstack((frame, separator,
                                               sparse, separator,
                                               sparse_filtered, separator,
                                               sparse_thr, separator,
                                               sparse_cc))
                    frame_queue.save_frame_to_file(load_directory,
                                                   frame=sparse_frames,
                                                   index=queue_center,
                                                   folder_name=args.custom_dir,
                                                   scale=400)

                    # TODO: Finish pipeline before tweaking stages further.
                    # Below are unused, but potentially viable processing
                    # stages. Don't touch these until you have the structure in
                    # place to properly evaluate the tweaks you're making!!!!

                    # BACKPROJECTION OR SOMETHING OR OTHER
                    # _, mask = cv2.threshold(sparse_filtered,
                    #                         thresh=0,
                    #                         maxval=255,
                    #                         type=(cv2.THRESH_BINARY+
                    #                               cv2.THRESH_OTSU))
                    # mask[mask == 255] = 1
                    # sparse_thresh = np.multiply(sparse_uint8, mask)

                    # GRAYSCALE MORPHOLOGICAL OPERATIONS
                    # for i in range(5):
                    #     sparse_opened = img.grey_opening(sparse_filtered,
                    #                                      size=(2, 2))
                    # sparse_eroded = img.grey_erosion(sparse_opened,
                    #                                  size=(1, 1))
                    #
                    # sparse_opened = \
                    #     img.grey_opening(sparse_thr, size=(2, 2)) \
                    #     .astype(sparse_thr.dtype)
                    # sparse_closed = \
                    #    img.grey_closing(sparse_opened, size=(2, 2)) \
                    #     .astype(sparse_thr.dtype)

            load_index += (1 + frame_queue.delay)
            if frame_queue.frames_read % 50 == 0:
                print("{0}/{1} frames processed."
                      .format(frame_queue.frames_read, total_frames))

        # Calculate error for total bird counts
        ground_truth = np.c_[ground_truth,
                             bird_count.reshape(-1, 1).astype(np.int)]
        error = ground_truth[:, 1] - ground_truth[:, 6]
        error_less = sum(error[error > 0])
        error_more = -1* sum(error[error < 0])

    breakpoint()


if __name__ == "__main__":
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
                        default=([16150, 300])
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
                        default="Test Folder"
                        )
    parser.add_argument("-p",
                        "--crop",
                        help="Corner coordinates for cropping.",
                        default=[(760, 650), (921, 686)]
                        )
    arguments = parser.parse_args()

    main(arguments)
