import swiftwatcher.process_video as pv
import os
import argparse as ap
import utils.cm as cm
import cv2
import numpy as np
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

        while frame_queue.frames_read < total_frames:
            # Load frame with specified index into FrameQueue object
            success = frame_queue.load_frame_from_file(load_directory,
                                                       load_index)

            # Process frame (grayscale, segmentation, etc.)
            if success:
                # Preprocessing
                frame_queue.convert_grayscale()
                frame_queue.crop_frame(corners=args.crop)
                frame_queue.frame_to_column()

                if frame_queue.frames_read > queue_center:
                    # Choosing index such that RPCA will use adjacent frames
                    # (forward and backwards) to "queue_center" frame
                    lowrank, sparse = \
                        frame_queue.rpca_decomposition(index=queue_center,
                                                       darker_only=True)

                    # Apply bilateral filter to remove noise, retain birds
                    sparse_filtered = sparse
                    for i in range(2):
                        sparse_filtered = cv2.bilateralFilter(sparse_filtered,
                                                              d=7,
                                                              sigmaColor=15,
                                                              sigmaSpace=1)

                    # Apply grayscale opening
                    # for i in range(5):
                    #     sparse_opened = img.grey_opening(sparse_filtered,
                    #                                      size=(2, 2))
                    # sparse_eroded = img.grey_erosion(sparse_opened,
                    #                                  size=(1, 1))

                    _, sparse_thr = cv2.threshold(sparse_filtered,
                                                  thresh=35,
                                                  maxval=255,
                                                  type=cv2.THRESH_TOZERO)
                    element = np.ones((2, 2))
                    sparse_opened = \
                        img.binary_opening(sparse_thr, structure=element) \
                        .astype(sparse_thr.dtype)

                    # Threshold result of bilateral filter, then backproject
                    # onto motion estimated image.
                    # _, mask = cv2.threshold(sparse_filtered,
                    #                         thresh=0,
                    #                         maxval=255,
                    #                         type=(cv2.THRESH_BINARY+
                    #                               cv2.THRESH_OTSU))
                    # mask[mask == 255] = 1
                    # sparse_thresh = np.multiply(sparse_uint8, mask)

                    num_components, sparse_cc = \
                        cv2.connectedComponents(sparse_opened, connectivity=4)
                    if num_components > 0:
                        sparse_cc = sparse_cc*(255/num_components)

                    # Reshape column vector form of image into normal frame
                    frame = np.reshape(frame_queue.queue[queue_center],
                                       (frame_queue.height, frame_queue.width))
                    separator = 255*np.ones(shape=(1, frame_queue.width),
                                            dtype=np.uint8)
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

            load_index += (1 + frame_queue.delay)
            if frame_queue.frames_read % 50 == 0:
                print("{0}/{1} frames processed."
                      .format(frame_queue.frames_read, total_frames))


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
                        default=([16200, 200])
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
                        default="Argument testing"
                        )
    parser.add_argument("-p",
                        "--crop",
                        help="Corner coordinates for cropping.",
                        default=[(760, 650), (921, 686)]
                        )
    arguments = parser.parse_args()

    main(arguments)
