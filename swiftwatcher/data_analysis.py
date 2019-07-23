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
import cv2
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


def event_comparison(df_eventinfo, df_groundtruth):
    """Generate dataframe comparing events in df_eventinfo with frame
    counts in df_groundtruth."""
    df_groundtruth = df_groundtruth[df_groundtruth["EXT_CHM"] > 0]
    df_eventinfo_cp = df_eventinfo.copy()
    df_eventinfo_cp["EXT_CHM"] = None
    df_combined = df_eventinfo.combine_first(df_groundtruth)
    df_combined["EXT_CHM"] = df_combined["EXT_CHM"].fillna(0)

    return df_combined


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
                                               bins=[-180, -135, -55, 180],
                                               labels=False)]

    return df_labels


def import_dataframes(args, df_list):

    if df_list == ["groundtruth"]:
        load_directory = args.default_dir
    else:
        load_directory = args.default_dir+args.custom_dir+"results/df-export/"

    dfs = {}
    for df_name in df_list:
        dfs[df_name] = pd.read_csv(load_directory+df_name+".csv")
        dfs[df_name]["TMSTAMP"] = pd.to_datetime(dfs[df_name]["TMSTAMP"])
        dfs[df_name]["TMSTAMP"] = dfs[df_name]["TMSTAMP"].dt.round('us')
        dfs[df_name].set_index(["TMSTAMP", "FRM_NUM"], inplace=True)

    return dfs


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


        # Cap groundtruth to specified frame range (done because sometimes I've
        # tested with a subset of frames, rather than the entire video.)
        index_less = gt_unprocessed[
            gt_unprocessed.index.levels[1] < args.load[0]].index
        index_more = gt_unprocessed[
            gt_unprocessed.index.levels[1] > args.load[1]].index
        gt_unprocessed.drop(index_less, inplace=True)
        gt_unprocessed.drop(index_more, inplace=True)
        gt_nonzero = gt_unprocessed[gt_unprocessed["EXT_CHM"] > 0]

        # Re-index groundtruth and predictions to have shared set of indexes
        union_index = pred_nonzero.index.union(gt_nonzero.index)
        gt_processed = gt_nonzero.reindex(index=union_index, fill_value=0)
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


def feature_engineering(args, df_comparison):
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


def train_classifier(args, params, df_eventinfo, df_groundtruth):
    def generate_blank_img():
        fq = FrameQueue(args, queue_size=params["queue_size"])
        width = int(fq.nn_region[1][0] - fq.nn_region[0][0])
        height = int(fq.nn_region[1][1] - fq.nn_region[0][1])

        return 127*np.ones((height, width))

    def generate_event_dfs():
        nonlocal df_eventinfo

        df_eventinfo["EXT_CHM"] = None
        df_eventinfo = df_eventinfo.combine_first(df_groundtruth)
        df_eventinfo["EXT_CHM"] = df_eventinfo["EXT_CHM"].fillna(0)
        df_eventinfo = df_eventinfo.dropna()

        positives = df_eventinfo.loc[df_eventinfo["EXT_CHM"].isin([1, 2, 3])]
        positives = positives.drop(columns="EXT_CHM")
        negatives = df_eventinfo.loc[df_eventinfo["EXT_CHM"].isin([0])]
        negatives = negatives.drop(columns="EXT_CHM")

        return positives, negatives

    def draw_centroids(img, centroid_list):
        counter = 0
        for centroid in centroid_list:
            centroid = (int(centroid[1]), int(centroid[0]))
            if counter == 0:
                prev = centroid
                pass

            cv2.line(img, centroid, prev, 255, 2)
            prev = centroid
            counter += 1

        return img

    def save_nn_images(positives, negatives):
        counter = 0
        for index, row in positives.iterrows():
            centroid_img = draw_centroids(np.copy(blank_img),
                                          literal_eval(row["CENTRDS"]))
            # centroid_img = cv2.resize(centroid_img, (224, 224))
            cv2.imwrite(save_directory + "1_{}.png".format(counter),
                        centroid_img)
            counter += 1

        counter = 0
        for index, row in negatives.iterrows():
            centroid_img = draw_centroids(np.copy(blank_img),
                                          literal_eval(row["CENTRDS"]))
            # centroid_img = cv2.resize(centroid_img, (224, 224))
            cv2.imwrite(save_directory + "0_{}.png".format(counter),
                        centroid_img)
            counter += 1

    # Create save directory if it does not already exist
    save_directory = args.default_dir+"NN/lines/"
    if not os.path.isdir(save_directory):
        try:
            os.makedirs(save_directory)
        except OSError:
            print("[!] Creation of the directory {0} failed."
                  .format(save_directory))

    blank_img = generate_blank_img()
    df_tp, df_tn = generate_event_dfs()
    save_nn_images(df_tp, df_tn)



