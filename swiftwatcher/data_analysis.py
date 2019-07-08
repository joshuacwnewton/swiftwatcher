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


def plot_result(args, df_groundtruth, df_estimation, key, flag):
    """Plot comparisons between estimation and ground truth for segments."""
    save_directory = args.default_dir + args.custom_dir + "results/plots/"
    if not os.path.isdir(save_directory):
        try:
            os.makedirs(save_directory)
        except OSError:
            print("[!] Creation of the directory {0} failed."
                  .format(save_directory))

    # Extract segment counts as series objects (from dataframes)
    es_series = df_estimation[key]
    gt_series = df_groundtruth[key]

    # Replace timestamp indices with frame number indices for visual clarity
    es_series.index = df_estimation["FRM_NUM"]
    gt_series.index = df_groundtruth["FRM_NUM"]

    # Calculate the overestimate error and underestimate error
    difference = es_series.subtract(gt_series)
    false_negatives = difference.where(difference > 0, 0)
    false_positives = -1 * difference.where(difference < 0, 0)

    # Initialize empty values
    series_plots = []
    legend = ""
    title = ""
    xlabel = ""
    ylabel = ""
    fig, ax = plt.subplots()

    # Set plot variables depending on flag passed
    if flag is "cumu_comparison":
        series_plots.append(es_series.cumsum())
        series_plots.append(gt_series.cumsum())
        legend = ["Estimation", "Ground Truth"]
        title = "Comparison between Cumulative Segment Sums"
        xlabel = "Frame Number"
        ylabel = "Segment Count"

    if flag is "false_positives":
        series_plots.append(false_positives.cumsum())
        series_plots.append(false_positives.rolling(50).sum())
        legend = ["Cumulative Sum", "Rolling Counts"]
        title = "False Positive Error for {}".format(key)
        xlabel = "Frame Number"
        ylabel = "False Positives"

    if flag is "false_negatives":
        series_plots.append(false_negatives.cumsum())
        series_plots.append(false_negatives.rolling(50).sum())
        legend = ["Cumulative Sum", "Rolling Counts"]
        title = "False Negative Error for {}".format(key)
        xlabel = "Frame Number"
        ylabel = "False Negatives"

    # Create and save plot
    for series in series_plots:
        series.plot(ax=ax)
    ax.legend(legend)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Text box with error count in corner of plot
    # First two values refer to position from specific x-axis/y-axis values
    # -- create something independent of data for placement?
    # ax2.text(7500, 1150, 'Total = {} Errors'.format(over_error.sum()),
    #          bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 5})
    plt.savefig(save_directory + '{0}_{1}.png'.format(key, flag),
                bbox_inches='tight')