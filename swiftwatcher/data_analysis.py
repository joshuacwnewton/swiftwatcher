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


def format_dataframes(df_estimation, df_groundtruth):
    # Parse TMSTAMP as datetime
    df_groundtruth["TMSTAMP"] = pd.to_datetime(df_groundtruth["TMSTAMP"])
    df_estimation["TMSTAMP"] = pd.to_datetime(df_estimation["TMSTAMP"])

    # Round DateTimeArray indices to microsecond precision
    df_groundtruth["TMSTAMP"] = df_groundtruth["TMSTAMP"].dt.round('us')
    df_estimation["TMSTAMP"] = df_estimation["TMSTAMP"].dt.round('us')

    # Set MultiIndex using both timestamps and framenumbers
    df_estimation.set_index(["TMSTAMP", "FRM_NUM"], inplace=True)
    df_groundtruth.set_index(["TMSTAMP", "FRM_NUM"], inplace=True)

    # Keep only the groundtruth counts which are present in estimates (rows)
    df_groundtruth = df_groundtruth.reindex(df_estimation.index)
    # Keep only the estimated counts which are present in groundtruth (columns)
    df_estimation_i = df_estimation[[c for c in df_groundtruth.columns]].copy()

    # Add frame information back to df_estimation
    df_estimation_i["FRMINFO"] = df_estimation["FRMINFO"]

    return df_estimation_i, df_groundtruth


def save_test_results(args, df_groundtruth, df_estimation):
    """Save the bird count estimations from image processing to csv files."""

    # Create save directory if it does not already exist
    save_directory = args.default_dir+args.custom_dir+"results/"
    if not os.path.isdir(save_directory):
        try:
            os.makedirs(save_directory)
        except OSError:
            print("[!] Creation of the directory {0} failed."
                  .format(save_directory))

    print("[*] Saving results of test to files.")

    # Using columns :-1 to exclude the "FRMINFO" column in df_estimation
    error_full = df_estimation.values[:, :-1] - df_groundtruth.values[:, :]
    correct = np.minimum(df_estimation.values[:, :-1],
                         df_groundtruth.values[:, :])

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
    df_results_cs["FRMINFO"] = df_estimation["FRMINFO"]
    df_results_sum = df_results.sum()
    df_results_err = df_results.copy()
    df_results_err["FRMINFO"] = df_estimation["FRMINFO"]
    df_results_err = df_results_err.loc[(df_results['total_error'] > 0)]
    df_results_tp = df_results.loc[(df_results['true_positives'] > 0)]
    df_results_md = df_results.loc[(df_results['missed_detections'] < 0)]
    df_results_fp = df_results.loc[(df_results['false_positives'] > 0)]

    # Writing the full estimation and summary of results to files
    df_estimation.to_csv(save_directory+"estimation.csv")
    df_groundtruth.to_csv(save_directory+"groundtruth.csv")
    df_results.to_csv(save_directory+"results_full.csv")
    df_results_cs.to_csv(save_directory+"results_cumulative.csv")
    df_results_sum.to_csv(save_directory+"results_summary.csv", header=False)
    df_results_md.to_csv(save_directory+"loc_missed-detections.csv")
    df_results_fp.to_csv(save_directory+"loc_false-positives.csv")
    df_results_tp.to_csv(save_directory+"loc_true-positives.csv")
    df_results_err.to_csv(save_directory+"error_information.csv")

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

    # Reset index to just "FRM_NUM"
    df_groundtruth = df_groundtruth.reset_index("TMSTAMP")
    df_estimation = df_estimation.reset_index("TMSTAMP")

    # Extract segment counts as series objects (from dataframes)
    es_series = df_estimation[key]
    gt_series = df_groundtruth[key]

    # Calculate the overestimate error and underestimate error
    difference = es_series.subtract(gt_series)
    false_positives = difference.where(difference > 0, 0)
    false_negatives = -1 * difference.where(difference < 0, 0)

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
    ax.legend(legend, loc="upper left")
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
