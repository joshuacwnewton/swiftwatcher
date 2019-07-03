# Stdlib imports
import os
import csv

# Data science libraries
import numpy as np
import pandas as pd

# Data visualization libraries
import matplotlib.pyplot as plt

# Needed to fetch video parameters for generating empty groundtruth file
from swiftwatcher.video_processing import FrameQueue
# import seaborn
# seaborn.set()


def empty_gt_generator(args):
    """Helper function for generating an empty file to store manual
    ground truth annotations. Ensures formatting is consistent."""

    # Create save directory if it does not already exist
    gt_dir = args.groundtruth.split("/")[0]
    save_directory = args.default_dir+gt_dir+"/"
    if not os.path.isdir(save_directory):
        try:
            os.makedirs(save_directory)
        except OSError:
            print("[!] Creation of the directory {0} failed."
                  .format(save_directory))

    # Create Series of DateTimeIndex indices (i.e. frame timestamps)
    frame_queue = FrameQueue(args)
    nano = (1/frame_queue.fps) * 1e9
    frame_queue.stream.release()  # Not needed once fps is extracted
    num_timestamps = args.load[1] - args.load[0]
    duration = (num_timestamps - 1) * nano
    indices = pd.date_range(start=args.timestamp,
                            end=(pd.Timestamp(args.timestamp) +
                                 pd.Timedelta(duration, 'ns')),
                            periods=num_timestamps)

    # Create a Series of frame numbers which correspond to the timestamps
    framenumbers = np.array(range(num_timestamps))

    # Create an empty DataFrame for ground truth annotations to be put into
    df_empty = pd.DataFrame(framenumbers, index=indices)
    df_empty.index.rename("TMSTAMP", inplace=True)
    df_empty.columns = ["FRM_NUM"]

    # Save for editing in Excel/LibreOffice Calc/etc.
    df_empty.to_csv(save_directory+"empty-groundtruth.csv")


def save_test_config(args, params):
    """Save the parameters chosen for the given test of the algorithm. Some
    parameters include commas, so files are delimited with semicolons."""
    save_directory = args.default_dir+args.custom_dir
    if not os.path.isdir(save_directory):
        try:
            os.makedirs(save_directory)
        except OSError:
            print("[!] Creation of the directory {0} failed."
                  .format(save_directory))

    # Writing a summary of the parameters to a file
    with open(save_directory + "/parameters.csv", 'w') as csv_file:
        filewriter = csv.writer(csv_file, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for key in params.keys():
            filewriter.writerow(["{}".format(key),
                                 "{}".format(params[key])])


def save_test_results(args, df_groundtruth, df_estimation):
    """Save the bird count estimations from image processing to csv files.

    Count labels:
        0, frame_number
        1, total_birds
        2, total_matches
        3, appeared_from_chimney
        4, appeared_from_edge
        5, appeared_ambiguous (could be removed, future-proofing for now)
        6, disappeared_to_chimney
        7, disappeared_to_edge
        8, disappeared_ambiguous (could be removed, future-proofing for now)
        9, outlier_behavior

    Estimate array contains a 10th catch-all count, "segmentation_error"."""

    # Create save directory if it does not already exist
    save_directory = args.default_dir+args.custom_dir+"results/"
    if not os.path.isdir(save_directory):
        try:
            os.makedirs(save_directory)
        except OSError:
            print("[!] Creation of the directory {0} failed."
                  .format(save_directory))

    print("[========================================================]")
    print("[*] Saving results of test to files.")

    # Round DateTimeArray indices to microsecond precision
    df_groundtruth.index = df_groundtruth.index.round('us')
    df_estimation.index = df_estimation.index.round('us')

    # Keep only the estimated counts which are present in groundtruth (columns)
    df_estimation = df_estimation[[c for c in df_groundtruth.columns]].copy()
    # Keep only the groundtruth counts which are present in estimates (rows)
    df_groundtruth = df_groundtruth.loc[df_estimation.index]

    # Using columns 1:10 so that the "frame number" column is excluded
    error_full = df_estimation.values[:, 1:] - df_groundtruth.values[:, 1:]
    correct = np.minimum(df_estimation.values[:, 1:],
                         df_groundtruth.values[:, 1:])

    # Summarizing the performance of the algorithm across all frames
    results_summary = {
        # Commented out because new ground truth does not yet have full
        # segmentation counts.
        # "count_true": np.sum(ground_truth[0:num_counts, 1:10], axis=0),
        # "count_estimated": np.sum(count_estimate[:, 1:10], axis=0),
        "true_positives": correct.reshape((-1,)),
        "false_positives": np.maximum(error_full, 0).reshape((-1,)),
        "missed_detections": np.minimum(error_full, 0).reshape((-1,)),
        "total_error": abs(error_full).reshape((-1,)),
        "net_error": error_full.reshape((-1,))
    }
    df_results = pd.DataFrame(results_summary, index=df_groundtruth.index)

    # Generate alternate versions for visual clarity
    df_results_cs = df_results.cumsum()
    df_results_sum = df_results.sum()

    # Writing the full estimation and summary of results to files
    df_estimation.to_csv(save_directory+"estimation.csv")
    df_groundtruth.to_csv(save_directory+"groundtruth.csv")
    df_results.to_csv(save_directory+"results_full.csv")
    df_results_cs.to_csv(save_directory+"results_cumulative.csv")
    df_results_sum.to_csv(save_directory+"results_summary.csv", header=False)

    print("[-] Results successfully saved to files.")

    return df_results


def plot_segmentation_results(args, df_estimation, df_groundtruth):
    """Plot comparisons between estimation and ground truth for segments."""
    save_directory = args.default_dir + args.custom_dir + "results/"

    # Extract segment counts as series objects (from dataframes)
    segments_es = df_estimation["SEGMNTS"]
    segments_gt = df_groundtruth["SEGMNTS"]

    # Replace timestamp indices with frame number indices for visual clarity
    segments_es.index = df_estimation["FRM_NUM"]
    segments_gt.index = df_groundtruth["FRM_NUM"]

    # Calculate the overestimate error and underestimate error
    difference = segments_es.subtract(segments_gt)
    over_error = difference.where(difference > 0, 0)
    under_error = -1 * difference.where(difference < 0, 0)

    # Plot a comparison between the cumulative sums of both segment counts
    fig1, ax1 = plt.subplots()
    segments_es_cumu = segments_es.cumsum()
    segments_gt_cumu = segments_gt.cumsum()
    segments_es_cumu.plot(ax=ax1)
    segments_gt_cumu.plot(ax=ax1)
    ax1.legend(["ESTIMATION", "GROUND TRUTH"])
    ax1.set_title('Comparison between Cumulative Segment Sums')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Segment Count')
    plt.savefig(save_directory+'segment_totals.png', bbox_inches='tight')

    # Plot a comparison between the cumulative sums of both error types
    fig2, ax2 = plt.subplots()
    over_cumu = over_error.cumsum()
    rolling_over = over_error.rolling(50).sum()
    ax2.text(7500, 1150, 'Total = {} Errors'.format(over_error.sum()),
             bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 5})
    over_cumu.plot(ax=ax2, color="black", linestyle="--")
    rolling_over.plot(ax=ax2, color="black")
    ax2.legend(["Cumulative Sum", "Rolling Counts"])
    ax2.set_title('Overestimate Error for Swift Segments')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Error Count')
    ax2.axis([7200, 16200, 0, 1250])
    plt.savefig(save_directory+'error_over.png', bbox_inches='tight')

    # Plot a comparison between the rolling sums of both error types
    fig3, ax3 = plt.subplots()
    under_cumu = under_error.cumsum()
    rolling_under = under_error.rolling(50).sum()
    ax3.text(7500, 1150, 'Total = {} Errors'.format(under_error.sum()),
             bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 5})
    under_cumu.plot(ax=ax3, color="black", linestyle="--")
    rolling_under.plot(ax=ax3, color="black")
    ax3.legend(["Cumulative Sum", "Rolling Counts"])
    ax3.set_title('Underestimate Error for Swift Segments')
    ax3.set_xlabel('Frame Number')
    ax3.set_ylabel('Error Count')
    ax3.axis([7200, 16200, 0, 1250])
    plt.savefig(save_directory+'error_under.png', bbox_inches='tight')
