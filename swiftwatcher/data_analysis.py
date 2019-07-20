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


def format_dataframes(args, df_groundtruth, df_eventinfo):
    # Cap groundtruth to specified frame range (done because sometimes I've
    # tested with a subset of frames, rather than the entire video.)
    index_less = df_groundtruth[df_groundtruth['FRM_NUM'] < args.load[0]].index
    index_more = df_groundtruth[df_groundtruth['FRM_NUM'] > args.load[1]].index
    df_groundtruth.drop(index_less, inplace=True)
    df_groundtruth.drop(index_more, inplace=True)

    # Parse TMSTAMP as datetime
    df_groundtruth["TMSTAMP"] = pd.to_datetime(df_groundtruth["TMSTAMP"])
    df_eventinfo["TMSTAMP"] = pd.to_datetime(df_eventinfo["TMSTAMP"])

    # Round DateTimeArray indices to microsecond precision (to prevent rounding
    # errors from the (default) nanosecond precision, which isn't necessary.)
    df_groundtruth["TMSTAMP"] = df_groundtruth["TMSTAMP"].dt.round('us')
    df_eventinfo["TMSTAMP"] = df_eventinfo["TMSTAMP"].dt.round('us')

    # Set MultiIndex using both timestamps and framenumbers
    df_groundtruth.set_index(["TMSTAMP", "FRM_NUM"], inplace=True)
    df_eventinfo.set_index(["TMSTAMP", "FRM_NUM"], inplace=True)

    return df_groundtruth, df_eventinfo


def generate_feature_vectors(df_eventinfo):
    """Use segment information to generate feature vectors for each event."""

    def compute_angle(centroid_list):
        # If loading from csv, convert from str to list
        if type(centroid_list) is str:
            centroid_list = literal_eval(centroid_list)

        del_y = centroid_list[0][0] - centroid_list[-1][0]
        del_x = -1 * (centroid_list[0][1] - centroid_list[-1][1])
        angle = math.degrees(math.atan2(del_y, del_x))

        return angle

    df_features = pd.DataFrame(index=df_eventinfo.index)

    df_features["ANGLE"] = df_eventinfo.apply(
        lambda row: compute_angle(row["CENTRDS"]),
        axis=1
    )

    return df_features


def classify_feature_vectors(df_features):
    """Classify "segment disappeared" events based on associated feature
    vectors.

    Note: currently this is done using a hard-coded values, but
    if time permits I would like to transition to a ML classifier."""

    df_labels = pd.DataFrame(index=df_features.index)

    df_labels["EXT_CHM"] = np.array([0, 1, 0])[pd.cut(df_features["ANGLE"],
                                               bins=[-180, -120, -35, 180],
                                               labels=False)]

    return df_labels


def export_dataframes(args, dataframe_dict):
    # Create save directory if it does not already exist
    save_directory = args.default_dir+args.custom_dir+"results/df-export/"
    if not os.path.isdir(save_directory):
        try:
            os.makedirs(save_directory)
        except OSError:
            print("[!] Creation of the directory {0} failed."
                  .format(save_directory))

    for key, value in dataframe_dict.items():
        value.to_csv(save_directory+"{}.csv".format(key))


def evaluate_results(args, df_groundtruth, df_prediction):
    """Save the bird count estimations from image processing to csv files."""

    def prepare_dataframes(pred_unprocessed, gt_unprocessed):
        # Merge multiple predicted events from same frame into single row
        pred_merged = pred_unprocessed.reset_index().groupby(['TMSTAMP',
                                                              'FRM_NUM']).sum()
        pred_nonzero = pred_merged.loc[(pred_merged['EXT_CHM'] > 0)]

        # Re-index groundtruth and predictions to have shared set of indexes
        union_index = pred_nonzero.index.union(gt_unprocessed.index)
        gt_processed = gt_unprocessed.reindex(index=union_index, fill_value=0)
        pred_processed = pred_nonzero.reindex(index=union_index, fill_value=0)

        return pred_processed, gt_processed

    def calculate_metrics(pred, pred_full, gt, gt_full):
        missed_event_loc = gt.index.difference(pred.index)
        missed_events = gt.loc[missed_event_loc, :]["EXT_CHM"].sum()
        error_full = np.subtract(pred_full.values, gt_full.values)
        correct = np.minimum(pred_full.values, gt_full.values)

        metrics = {
            "true_positives": correct.reshape((-1,)),
            "false_positives": np.maximum(error_full, 0).reshape((-1,)),
            "false_negatives": np.minimum(error_full, 0).reshape((-1,)),
            "abs_error": abs(error_full).reshape((-1,)),
            "net_error": error_full.reshape((-1,))
        }

        metrics = pd.DataFrame(metrics, index=gt_full.index)
        # These "metrics_totals" calculates are a bit messy, and I would like
        # to take some time to make it a bit clearer. Having different counts
        # at different stages of the algorithm (segmentation, classification)
        # can be a bit hard to track.
        metric_totals = {
            "te": pred["EXT_CHM"].size,
            "pp": pred["EXT_CHM"].sum(),
            "pn": pred["EXT_CHM"].size - pred["EXT_CHM"].sum(),
            "ap": gt["EXT_CHM"].sum(),
            "me": missed_events,
            "tp": metrics["true_positives"].sum(),
            "fp": metrics["false_positives"].sum(),
            "tn": pred["EXT_CHM"].size-(metrics["true_positives"].sum() +
                                        metrics["false_positives"].sum() +
                                        (-1*metrics["false_negatives"].sum()
                                         - missed_events)),
            "fn": -1*metrics["false_negatives"].sum() - missed_events,
            "md": -1*metrics["false_negatives"].sum(),
            "ae": metrics["abs_error"].sum(),
            "ne": metrics["net_error"].sum(),
        }

        return metric_totals

    def save_evaluation(save_directory, totals):
        if not os.path.isdir(save_directory):
            try:
                os.makedirs(save_directory)
            except OSError:
                print("[!] Creation of the directory {0} failed."
                      .format(save_directory))

        results = [
            "EVENT DETECTION\n",
            "   -{} possible swifts to detect.\n".format(totals["ap"]),
            "   -{}/{} swifts were detected.\n".format(totals["ap"] -
                                                       totals["me"],
                                                       totals["ap"]),
            "   -{}/{} swifts were missed entirely.".format(totals["me"],
                                                            totals["ap"]),
            " (Due to poor matching, overlapping, etc.)\n"
            "EVENT CLASSIFICATION\n",
            "   -{} events were detected by"
            " segmentation/matching.\n".format(totals["te"]),
            "   -{}/{} events labeled as positives.\n".format(totals["pp"],
                                                              totals["te"]),
            "       -{}/{} labeled positives were TPs.\n".format(totals["tp"],
                                                                 totals["pp"]),
            "       -{}/{} labeled positives were FPs.\n".format(totals["fp"],
                                                                 totals["pp"]),
            "   -{}/{} events were labeled negatives.\n".format(totals["pn"],
                                                                totals["te"]),
            "       -{}/{} labeled negatives were TNs.\n".format(totals["tn"],
                                                                 totals["pn"]),
            "       -{}/{} labeled negatives were FNs.\n".format(totals["fn"],
                                                                 totals["pn"]),
            "FINAL EVALUATION\n",
            "   -Precision: {}\n".format(round(totals["tp"] /
                                         (totals["tp"] + totals["fp"]), 4)),
            "   -Recall: {}\n".format(round(totals["tp"] /
                                      (totals["tp"] + totals["md"]), 4)),
            "   -Accuracy: {}\n".format(round((totals["tp"] + totals["tn"]) /
                                        (totals["tp"] + totals["tn"] +
                                         totals["fp"] + totals["md"]), 4))
        ]

        file = open(save_directory+'results.txt', 'w')
        file.writelines(results)
        file.close()

    df_prediction_full, df_groundtruth_full = prepare_dataframes(df_prediction,
                                                                 df_groundtruth)
    dict_totals = calculate_metrics(df_prediction, df_prediction_full,
                                    df_groundtruth, df_groundtruth_full)
    save_evaluation(args.default_dir+args.custom_dir+"results/", dict_totals)


def plot_result(args, df_groundtruth, df_prediction, key, flag):
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
    df_prediction = df_prediction.reset_index("TMSTAMP")

    # Extract segment counts as series objects (from dataframes)
    es_series = df_prediction[key]
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
    num_timestamps = args.load[1] - args.load[0] + 1
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


def feature_engineering(args):
    """Testing function for exploring different features."""

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

    save_directory = args.default_dir+args.custom_dir+"classification/"

    df_class = pd.read_csv(save_directory+"classifier-xy.csv")
    df_class["AVGDIST"] = df_class.apply(
        lambda row: compute_avg_distance(row["CENTRDS"]),
        axis=1
    )
    positives = df_class.loc[df_class["GTLABEL"].isin([1])]
    negatives = df_class.loc[df_class["GTLABEL"].isin([0])]

    ax = positives["ANGLE_2"].hist(bins=72, alpha=0.8)
    ax = negatives["ANGLE_2"].hist(bins=72, alpha=0.5)
    ax.legend(["'Into Chimney' Samples", "'Not Into Chimney' Samples"],
              loc="upper right")
    ax.set_title("Comparison in Flight Path Angle for Detected Segments")
    ax.set_xlabel("Angle (Degrees)")
    ax.set_ylabel("Total Segments")
    fig = ax.get_figure()
    fig.savefig(save_directory+'hist_angle.png')

    # import matplotlib.pyplot as plt
    # ax = positives["ANGLE_1"].hist(bins=72, alpha=0.8)
    # ax = negatives["ANGLE_1"].hist(bins=72, alpha=0.5)
    # fig = ax.get_figure()
    # fig.savefig(save_directory+'hist_angle1.png')
    #
    # plt.cla()
    #
    # ax = positives["ANGLE_3"].hist(bins=72, alpha=0.8)
    # ax = negatives["ANGLE_3"].hist(bins=72, alpha=0.5)
    # fig = ax.get_figure()
    # fig.savefig(save_directory+'hist_angle3.png')
    #
    # plt.cla()
    #
    # ax = positives["AVGDIST"].hist(bins=72, alpha=0.8)
    # ax = negatives["AVGDIST"].hist(bins=72, alpha=0.5)
    # fig = ax.get_figure()
    # fig.savefig(save_directory+'avgdist.png')
    #
    # plt.cla()
    #
    # ax = positives.plot.scatter(x='ANGLE_2', y='AVGDIST', color='Green', label='Positives')
    # negatives.plot.scatter(x='ANGLE_2', y='AVGDIST', color='Red', label='Negatives', ax=ax)
    # fig = ax.get_figure()
    # fig.savefig('scatter.png')

