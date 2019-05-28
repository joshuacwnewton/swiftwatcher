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

    # Code to load previously extracted frames and apply image processing to them
    if args.reuse:
        # Initialize parameters necessary to load previously extracted frames
        load_directory = (args.video_dir +  # Assumed convention from pv.extract_frames()
                          os.path.splitext(args.filename)[0])
        load_index = args.reuse[0]          # Index of frame to load next
        total_frames = args.reuse[1]        # Total number of frames to load and process

        # Create FrameQueue object
        queue_size = 20
        queue_center = int((queue_size-1)/2)
        frame_queue = pv.FrameQueue(args.video_dir, args.filename, queue_size=queue_size)
        frame_queue.stream.release()  # VideoCapture not needed if frames are being reused

        while frame_queue.frames_read < total_frames:
            # Load frame with specified index into FrameQueue object
            success = frame_queue.load_frame_from_file(load_directory, load_index)

            # Process frame (grayscale, segmentation, etc.)
            if success:
                # Preprocessing
                frame_queue.convert_grayscale()
                frame_queue.crop_frame(corners=[(745, 620), (920, 690)])  # top-left [w,h], bottom-right [w,h]
                frame_queue.frame_to_column()

                if frame_queue.frames_read > queue_center:
                    # Robust PCA using adjacent frames from forward and backwards in time
                    lowrank, sparse = frame_queue.rpca_decomposition(index=queue_center,
                                                                     darker_only=True)

                    # Apply bilateral filter to remove noise, retain birds
                    sparse_filtered = sparse
                    for i in range(2):
                        sparse_filtered = cv2.bilateralFilter(sparse_filtered, 7, 15, 1)

                    # Apply grayscale opening
                    #for i in range(5):
                    #     sparse_opened = img.grey_opening(sparse_filtered, size=(2, 2))
                    # sparse_eroded = img.grey_erosion(sparse_opened, size=(1, 1))

                    ret, sparse_thresh = cv2.threshold(sparse_filtered, 35, 255, cv2.THRESH_TOZERO)
                    sparse_opened = img.binary_opening(sparse_thresh,
                                                       structure=np.ones((2, 2))).astype(sparse_thresh.dtype)
                    # Threshold result of bilateral filter, backproject onto motion estimated image
                    # _, mask = cv2.threshold(sparse_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # mask[mask == 255] = 1
                    # sparse_thresh = np.multiply(sparse_uint8, mask)

                    retval, sparse_cc = cv2.connectedComponents(sparse_opened, connectivity=4)
                    if sparse_cc.max() is not 0:
                        sparse_cc = sparse_cc*(255/sparse_cc.max())

                    # Print 3 stages of filtering to compare
                    frame = np.reshape(frame_queue.queue[queue_center], (frame_queue.height, frame_queue.width))
                    sparse_frames = np.vstack((frame, sparse_thresh, sparse_cc))
                    frame_queue.save_frame_to_file(load_directory, frame=sparse_frames,
                                                   index=queue_center,
                                                   folder_name=args.custom_dir,
                                                   scale=400)
                    # Unused elements of RPCA
                    # frame_queue.save_frame_to_file(load_directory, frame=lowrank,
                    #                                index=queue_center,
                    #                                folder_name=args.custom_dir,
                    #                                file_suffix="b_RPCAlowrank", scale=400)
                    # sparse_cm = cm.apply_custom_colormap(sparse_abs, cmap="viridis")
                    # frame_queue.save_frame_to_file(load_directory, frame=sparse_cm,
                    #                                index=queue_center,
                    #                                folder_name=args.custom_dir,
                    #                                file_suffix="c_RPCAsparsecm", scale=400)

            load_index += (1 + frame_queue.delay)
            if frame_queue.frames_read % 50 == 0:
                print("{0}/{1} frames processed.".format(frame_queue.frames_read, total_frames))


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
                        help="Custom directory for extracted frame image files",
                        default="Unspecified Test Folder"
                        )
    arguments = parser.parse_args()

    main(arguments)
