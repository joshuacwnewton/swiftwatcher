# Stdlib imports
import os
import csv
import math
from ast import literal_eval

# Data science libraries
import numpy as np
import pandas as pd

# Data visualization libraries
import matplotlib.pyplot as plt

# Needed to fetch video parameters for generating empty groundtruth file
from swiftwatcher.video_processing import FrameQueue

# Classifier modules
from sklearn import svm
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


def format_dataframes(args, df_groundtruth, df_events):
    # Cap groundtruth to specified frame range
    index_less = df_groundtruth[df_groundtruth['FRM_NUM'] < args.load[0]].index
    index_more = df_groundtruth[df_groundtruth['FRM_NUM'] > args.load[1]].index
    df_groundtruth.drop(index_less, inplace=True)
    df_groundtruth.drop(index_more, inplace=True)

    # Remove ground truth rows with no instances of swifts entering
    index_zero = df_groundtruth[df_groundtruth['EXT_CHM'] == 0].index
    df_groundtruth.drop(index_zero, inplace=True)

    # Parse TMSTAMP as datetime
    df_groundtruth["TMSTAMP"] = pd.to_datetime(df_groundtruth["TMSTAMP"])
    df_events["TMSTAMP"] = pd.to_datetime(df_events["TMSTAMP"])

    # Round DateTimeArray indices to microsecond precision
    df_groundtruth["TMSTAMP"] = df_groundtruth["TMSTAMP"].dt.round('us')
    df_events["TMSTAMP"] = df_events["TMSTAMP"].dt.round('us')

    # Set MultiIndex using both timestamps and framenumbers
    df_groundtruth.set_index(["TMSTAMP", "FRM_NUM"], inplace=True)
    df_events.set_index(["TMSTAMP", "FRM_NUM"], inplace=True)

    return df_groundtruth, df_events


def generate_feature_vectors(df_events):
    """Use segment information to generate feature vectors for each event."""

    def compute_angle(centroid_list):
        # If loading from csv, convert from str to list
        if type(centroid_list) is str:
            centroid_list = literal_eval(centroid_list)

        del_y = centroid_list[0][0] - centroid_list[-1][0]
        del_x = -1 * (centroid_list[0][1] - centroid_list[-1][1])
        angle = math.degrees(math.atan2(del_y, del_x))

        return angle

    df_features = pd.DataFrame(index=df_events.index)
    df_features["ANGLE"] = df_events.apply(
        lambda row: compute_angle(row["CENTRDS"]),
        axis=1
    )

    return df_features


def train_classifier(args):

    def compute_avg_distance(centroid_list):
        # If loading from csv, convert from str to list
        if type(centroid_list) is str:
            centroid_list = literal_eval(centroid_list)

        dist_sum = 0
        for i in range(len(centroid_list) - 2):
            c1 = centroid_list[i+1]
            c2 = centroid_list[i+2]
            dist_sum += math.sqrt((c2[0] - c1[0])**2 + (c2[1] - c2[1])**2)
            avg_distance = dist_sum / (len(centroid_list) - 2)

        if dist_sum == 0:
            for i in range(len(centroid_list) - 1):
                c1 = centroid_list[i]
                c2 = centroid_list[i + 1]
                dist_sum += math.sqrt(
                    (c2[0] - c1[0]) ** 2 + (c2[1] - c2[1]) ** 2)
            avg_distance = dist_sum / (len(centroid_list) - 1)

        return avg_distance


    import matplotlib.pyplot as plt
    df_class = pd.read_csv(args.default_dir+"/groundtruth/classifier-xy.csv")
    df_class["AVGDIST"] = df_class.apply(
        lambda row: compute_avg_distance(row["CENTRDS"]),
        axis=1
    )
    positives = df_class.loc[df_class["GTLABEL"].isin([1])]
    negatives = df_class.loc[df_class["GTLABEL"].isin([0])]

    ax = positives["ANGLE_1"].hist(bins=72, alpha=0.8)
    ax = negatives["ANGLE_1"].hist(bins=72, alpha=0.5)
    fig = ax.get_figure()
    fig.savefig('hist_angle1.png')

    plt.cla()

    ax = positives["ANGLE_2"].hist(bins=72, alpha=0.8)
    ax = negatives["ANGLE_2"].hist(bins=72, alpha=0.5)
    fig = ax.get_figure()
    fig.savefig('hist_angle2.png')

    plt.cla()

    ax = positives["ANGLE_3"].hist(bins=72, alpha=0.8)
    ax = negatives["ANGLE_3"].hist(bins=72, alpha=0.5)
    fig = ax.get_figure()
    fig.savefig('hist_angle3.png')

    plt.cla()

    ax = positives["AVGDIST"].hist(bins=72, alpha=0.8)
    ax = negatives["AVGDIST"].hist(bins=72, alpha=0.5)
    fig = ax.get_figure()
    fig.savefig('avgdist.png')

    plt.cla()

    ax = positives.plot.scatter(x='ANGLE_2', y='AVGDIST', color='Green', label='Positives')
    negatives.plot.scatter(x='ANGLE_2', y='AVGDIST', color='Red', label='Negatives', ax=ax)
    fig = ax.get_figure()
    fig.savefig('scatter.png')

    # model = svm.SCV(gamma='auto')


def classify_feature_vectors(df_features):
    df_labels = pd.DataFrame(index=df_features.index)

    df_labels["LABEL"] = np.array([0,1,0])[pd.cut(df_features["ANGLE"],
                                                  bins=[-180, -125, -55, 180],
                                                  labels=False)]

    return df_labels


def generate_counts(df_labels):
    df_counts = df_labels.reset_index().groupby(['TMSTAMP', 'FRM_NUM']).sum()
    df_counts = df_counts.loc[(df_counts['LABEL'] > 0)]
    df_counts.columns = ["EXT_CHM"]

    return df_counts


def save_test_results(args, df_groundtruth, df_estimation):
    """Save the bird count estimations from image processing to csv files."""

    print("[*] Saving results of test to files.")

    # Create save directory if it does not already exist
    save_directory = args.default_dir+args.custom_dir+"results/"
    if not os.path.isdir(save_directory):
        try:
            os.makedirs(save_directory)
        except OSError:
            print("[!] Creation of the directory {0} failed."
                  .format(save_directory))

    # Using columns :-1 to exclude the "FRMINFO" column in df_estimation
    error_full = df_estimation.values[:, :] - df_groundtruth.values[:, :]
    correct = np.minimum(df_estimation.values[:, :],
                         df_groundtruth.values[:, :])

    results = {
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
    df_results = pd.DataFrame(results, index=df_groundtruth.index)
    df_results.to_csv(save_directory+"results_full.csv")

    df_estimation.to_csv(save_directory+"estimation.csv")
    df_groundtruth.to_csv(save_directory+"groundtruth.csv")

    df_results_cs = df_results.cumsum()
    df_results_cs.to_csv(save_directory + "results_cumulative.csv")

    df_results_sum = df_results.sum()
    df_results_sum["precision"] = 100 * (df_results_sum["true_positives"] /
                                         (df_results_sum["true_positives"] +
                                          df_results_sum["false_positives"]))
    df_results_sum["recall"] = 100 * (df_results_sum["true_positives"] /
                                      (df_results_sum["true_positives"] +
                                       -1*df_results_sum["missed_detections"]))
    df_results_sum.to_csv(save_directory + "results_summary.csv", header=False)

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
