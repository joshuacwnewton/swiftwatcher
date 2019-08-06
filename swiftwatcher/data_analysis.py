# Stdlib imports
import os
from os import fspath
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

# Needed for pairwise iteration
from itertools import tee


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

    if not df_eventinfo.empty:
        df_features = pd.DataFrame(index=df_eventinfo.index)
        df_features["ANGLE"] = df_eventinfo.apply(
            lambda row: compute_angle(row["CENTRDS"]),
            axis=1
        )
    else:
        df_features = df_eventinfo

    return df_features


def generate_classifications(df_features):
    """Classify "segment disappeared" events based on associated feature
    vectors.

    Note: currently this is done using a hard-coded values, but
    if time permits I would like to transition to a ML classifier."""

    if not df_features.empty:
        df_labels = pd.DataFrame(index=df_features.index)
        df_labels["ENTERPR"] = np.array([0, 1, 0])[pd.cut(df_features["ANGLE"],
                                                   bins=[-180, -125, -55, 180],
                                                   labels=False)]
        # Give each classified event a value of 1, so that when multiple events
        # on a single timestamp are merged, it will clearly show EVENTS = (>=2)
        df_labels["EVENTS"] = 1
    else:
        df_labels = df_features

    return df_labels


def generate_comparison(config, df_prediction, df_groundtruth):
    """Generate dataframe comparing events in df_eventinfo with frame
    counts in df_groundtruth."""
    def fix_offbyone(df_comparison):
        def pairwise(iterable):
            """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)

        df_comparison = df_comparison.fillna(0)

        rows_to_drop = []
        for (i1, row1), (i2, row2) in pairwise(df_comparison.iterrows()):
            if (type(i1) is tuple) and (type(i2) is tuple):
                i1 = i1[1]
                i2 = i2[1]

            if i2 - i1 == 1:
                diff1 = row1["ENTERGT"] - row1["ENTERPR"]
                diff2 = row2["ENTERGT"] - row2["ENTERPR"]

                # Condition for FN/MD and FP in sequential frames
                if (diff1 > 0) and (diff2 < 0):
                    # Shift "off-by-one" GT count to cancel out errors
                    offbyone = min(diff1, abs(diff2))
                    row1["ENTERGT"] -= offbyone
                    row2["ENTERGT"] += offbyone
                    # Remove row if empty (e.g. 1 0 0 -> 0 0 0 after shift)
                    if np.array_equal(row1.values, [0, 0, 0]):
                        rows_to_drop.append(i1)

                # Condition for FP and FN/MD in sequential frames
                elif (diff1 < 0) and (diff2 > 0):
                    # Shift "off-by-one" GT count to cancel out errors
                    offbyone = min(abs(diff1), diff2)
                    row2["ENTERGT"] -= offbyone
                    row1["ENTERGT"] += offbyone

                    # Remove row if empty (e.g. 1 0 0 -> 0 0 0 after shift)
                    if np.array_equal(row2.values, [0, 0, 0]):
                        rows_to_drop.append(i2)

        if type(df_comparison.index) == pd.MultiIndex:
            df_comparison_rm = df_comparison.drop(level=1, index=rows_to_drop)
        else:
            df_comparison_rm = df_comparison.drop(index=rows_to_drop)

        return df_comparison_rm

    if not df_prediction.empty:
        df_groundtruth = df_groundtruth[df_groundtruth["ENTERGT"] > 0]
        df_prediction_cp = df_prediction.copy()
        df_prediction_cp = df_prediction_cp.reset_index().groupby(['TMSTAMP',
                                                                  'FRM_NUM']).sum()
        df_prediction_cp["ENTERGT"] = None

        if "TMSTAMP" not in df_groundtruth.index.names:
            df_prediction_cp = df_prediction_cp.reset_index(level=[0])
            df_prediction_cp = df_prediction_cp.drop(columns="TMSTAMP")

        df_combined = df_prediction_cp.combine_first(df_groundtruth)
        df_combined["ENTERGT"] = df_combined["ENTERGT"].fillna(0)

        if type(df_combined.index) == pd.MultiIndex:
            indexes_to_drop = df_combined[(df_combined.index.levels[1]
                                           < config["start_frame"] - 1) |
                                          (df_combined.index.levels[1]
                                           > config["end_frame"] - 1)].index
        else:
            indexes_to_drop = df_combined[(df_combined.index
                                           < config["start_frame"] - 1) |
                                          (df_combined.index
                                           > config["end_frame"] - 1)].index
        df_combined_fixed = fix_offbyone(df_combined)
        df_combined_fixed = df_combined_fixed.drop(index=indexes_to_drop)

    else:
        df_combined_fixed = df_prediction

    return df_combined_fixed


def import_dataframes(load_directory, df_list):
    dfs = {}
    for df_name in df_list:
        dfs[df_name] = \
            pd.read_csv(fspath(load_directory/"{}.csv".format(df_name)))

        if "TMSTAMP" in dfs[df_name].columns:
            dfs[df_name]["TMSTAMP"] = pd.to_datetime(dfs[df_name]["TMSTAMP"])
            dfs[df_name]["TMSTAMP"] = dfs[df_name]["TMSTAMP"].dt.round('us'
                                                                       '')
            dfs[df_name].set_index(["TMSTAMP", "FRM_NUM"], inplace=True)
        else:
            dfs[df_name].set_index(["FRM_NUM"], inplace=True)

    return dfs


def export_dataframes(test_directory, dataframe_dict):
    save_directory = test_directory/"results"/"df-export"
    if not save_directory.exists():
        try:
            save_directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            print("[!] Creation of the directory {0} failed."
                  .format(save_directory))

    for key, value in dataframe_dict.items():
        value.to_csv(fspath(save_directory/"{}.csv".format(key)))


def evaluate_results(test_directory, df_comparison):
    """Save the bird count estimations from image processing to csv files."""

    def split_comparison(comparison):
        event_types = {}

        # A timestamp contains an event labeled 'positive' if the number of
        # predicted birds is nonzero.
        positives = comparison[comparison["ENTERPR"] > 0]
        # A timestamp contains a TP event if the ground truth count
        # (corresponding to a predicted count) is also nonzero.
        event_types["tp"] = positives[positives["ENTERGT"] > 0]
        # A timestamp contains a FP event if the predicted count is greater
        # than the corresponding groundtruth count.
        event_types["fp"] = positives[positives["ENTERPR"] >
                                      positives["ENTERGT"]]

        # A timestamp contains an event labeled 'negative' if the number of
        # predicted birds is less than the number of detected events.
        negatives = comparison[comparison["ENTERPR"] < comparison["EVENTS"]]
        # A timestamp contains a TN event if the number of events detected
        # is greater than the ground truth count.
        event_types["tn"] = negatives[negatives["EVENTS"] >
                                      negatives["ENTERGT"]]
        # A timestamp contains a FN event if predicted count is less than
        # the corresponding groundtruth count.
        event_types["fn"] = negatives[negatives["ENTERPR"] <
                                      negatives["ENTERGT"]]

        # A timestamp contains a missed detection if the number of detected
        # events is lower than the ground truth count.
        event_types["md"] = comparison[comparison["EVENTS"] <
                                       comparison["ENTERGT"]]

        return event_types

    def sum_counts(event_types, full_comparison):
        sums = {}

        sums["gt"] = int(np.sum(full_comparison["ENTERGT"]))
        sums["te"] = int(np.sum(full_comparison["EVENTS"]))
        sums["md"] = int(np.sum(np.subtract(event_types["md"]["ENTERGT"],
                                            event_types["md"]["EVENTS"])))

        # Breakdown of events labeled "positive"
        sums["p"] = int(np.sum(full_comparison["ENTERPR"]))
        sums["tp"] = int(np.sum(np.minimum(event_types["tp"]["ENTERPR"],
                                           event_types["tp"]["ENTERGT"])))
        sums["fp"] = sums["p"] - sums["tp"]

        # Breakdown of events labeled "negative"
        sums["n"] = int(np.sum(full_comparison["EVENTS"]) - sums["p"])
        sums["fn"] = (sums["gt"] - sums["md"]) - sums["tp"]
        sums["tn"] = sums["n"] - sums["fn"]

        # Sanity check to ensure calculated sums match total events
        assert sums["te"] == sums["tp"] + sums["fp"] + sums["tn"] + sums["fn"]

        sums["precision"] = 100 * round(sums["tp"] /
                                        (sums["tp"] + sums["fp"]), 4)
        sums["recall"] = 100 * round(sums["tp"] /
                                     (sums["tp"] + sums["fn"] + sums["md"]), 4)
        sums["accuracy"] = 100 * round((sums["tp"] + sums["tn"]) /
                                       (sums["tp"] + sums["tn"] + sums["fp"]
                                        + sums["fn"] + sums["md"]), 4)

        return sums

    def export_results(save_directory, sums):
        if not save_directory.exists():
            try:
                save_directory.mkdir(parents=True, exist_ok=True)
            except OSError:
                print("[!] Creation of the directory {0} failed."
                      .format(save_directory))

        results = [
            "EVENT DETECTION\n",
            "   -{} possible swifts to detect.\n".format(sums["gt"]),
            "   -{}/{} swifts were detected.\n".format(sums["gt"] -
                                                       sums["md"],
                                                       sums["gt"]),
            "   -{}/{} swifts were missed entirely.".format(sums["md"],
                                                            sums["gt"]),
            " (Due to poor matching, overlapping, etc.)\n",
            "\n",
            "EVENT CLASSIFICATION\n",
            "   -{} events were detected by segmentation/matching."
            "\n".format(sums["te"]),
            "   -{}/{} events labeled as positives.\n".format(sums["tp"] +
                                                              sums["fp"],
                                                              sums["te"]),
            "       -{}/{} labeled positives were TPs.\n".format(sums["tp"],
                                                                 sums["tp"] +
                                                                 sums["fp"]),
            "       -{}/{} labeled positives were FPs.\n".format(sums["fp"],
                                                                 sums["tp"] +
                                                                 sums["fp"]),
            "   -{}/{} events were labeled negatives.\n".format(sums["tn"] +
                                                                sums["fn"],
                                                                sums["te"]),
            "       -{}/{} labeled negatives were TNs.\n".format(sums["tn"],
                                                                 sums["tn"] +
                                                                 sums["fn"]),
            "       -{}/{} labeled negatives were FNs.\n".format(sums["fn"],
                                                                 sums["tn"] +
                                                                 sums["fn"]),
            "\n",
            "FINAL EVALUATION\n",
            "   -Precision = {} TPs / ({} TPs + {} FPs)\n"
            "              = {}%\n".format(sums["tp"], sums["tp"], sums["fp"],
                                           sums["precision"]),
            "   -Recall    = {} TPs / ({} TPs + {} FNs + {} MSs)\n"
            "              = {}%\n".format(sums["tp"], sums["tp"], sums["fn"],
                                           sums["md"], sums["recall"])
        ]

        file = open(fspath(save_directory/'results.txt'), 'w')
        file.writelines(results)
        file.close()

    if not df_comparison.empty:
        result_dict = split_comparison(df_comparison.copy(deep=True))
        sum_dict = sum_counts(result_dict, df_comparison)
        export_results(test_directory/"results", sum_dict)
    else:
        result_dict = {}

    return result_dict


def plot_result(test_directory, df_prediction, df_groundtruth=None, flag=None):
    """Plot comparisons between estimation and ground truth for segments."""
    save_directory = test_directory/"results"/"plots"
    if not save_directory.exists():
        try:
            save_directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            print("[!] Creation of the directory {0} failed."
                  .format(save_directory))

    if not df_prediction.empty:
        df_prediction = df_prediction.reset_index("TMSTAMP")
        es_series = df_prediction["ENTERPR"]

        if df_groundtruth is not None:
            if "TMSTAMP" in df_groundtruth.index.names:
                df_groundtruth = df_groundtruth.reset_index("TMSTAMP")
            gt_series = df_groundtruth["ENTERGT"]

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

        elif flag is "false_positives":
            series_plots.append(false_positives.cumsum())
            series_plots.append(false_positives.rolling(50).sum())
            legend = ["Cumulative Sum", "Rolling Counts"]
            title = "False Positive Error for Enter Chimney Count"
            xlabel = "Frame Number"
            ylabel = "False Positives"

        elif flag is "false_negatives":
            series_plots.append(false_negatives.cumsum())
            series_plots.append(false_negatives.rolling(50).sum())
            legend = ["Cumulative Sum", "Rolling Counts"]
            title = "False Negative Error for Enter Chimney Count"
            xlabel = "Frame Number"
            ylabel = "False Negatives"

        else:
            series_plots.append(es_series.cumsum())
            legend = ["Birds Entering Chimney"]
            title = "Timeline of Predicted Bird Counts"
            xlabel = "Frame Number"
            ylabel = "Segment Count"

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
        plt.savefig(fspath(save_directory/'{}.png'.format(flag)),
                    bbox_inches='tight')


# Functions that I only use on occasion and that can wait to be updated


def empty_gt_generator(args):
    """Helper function for generating an empty file to store manual
    ground truth annotations. Ensures formatting is consistent."""

    # Create save directory if it does not already exist
    save_directory = args.default_dir
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
    num_timestamps = frame_queue.total_frames
    duration = (num_timestamps - 1) * nano
    timestamps = pd.date_range(start=args.timestamp,
                              end=(pd.Timestamp(args.timestamp) +
                                   pd.Timedelta(duration, 'ns')),
                              periods=num_timestamps)
    timestamps = timestamps.round('us')

    # Create a Series of frame numbers which correspond to the timestamps
    framenumbers = np.array(range(num_timestamps))

    tuples = list(zip(timestamps, framenumbers))
    index = pd.MultiIndex.from_tuples(tuples, names=['TMSTAMP', 'FRM_NUM'])

    # Create an empty DataFrame for ground truth annotations to be put into
    df_empty = pd.DataFrame(index=index)
    df_empty["ENTERGT"] = 0
    df_empty["EXIT_GT"] = 0

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


def feature_engineering(args, result_dict):
    """Testing function for exploring different features."""

    def split_data():
        detected_events = None  # df_comparison.dropna()
        true_positives = detected_events[detected_events["EXT_CHM"] > 0]
        true_negatives = detected_events[detected_events["EXT_CHM"] == 0]

        return true_positives, true_negatives

    def visualize_path(positives, negatives):

        def generate_blank_img():
            fq = FrameQueue(args)

            blank = 127 * np.ones((fq.height, fq.width)).astype(np.uint8)
            blank_w_roi = cv2.addWeighted(fq.roi_mask, 0.25, blank, 0.75, 0)

            return blank_w_roi

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

        save_directory = (args.default_dir + args.custom_dir
                         + "feature-testing/" + "centroids/")
        if not os.path.isdir(save_directory):
            try:
                os.makedirs(save_directory)
            except OSError:
                print("[!] Creation of the directory {0} failed."
                      .format(save_directory))

        blank_img = generate_blank_img()

        counter = 0
        for index, row in positives.iterrows():
            if row["CENTRDS"] is str:
                centroid_img = draw_centroids(np.copy(blank_img),
                                              literal_eval(row["CENTRDS"]))
            else:
                centroid_img = draw_centroids(np.copy(blank_img),
                                              row["CENTRDS"])
            # centroid_img = cv2.resize(centroid_img, (224, 224))
            cv2.imwrite(save_directory + "1_{}.png".format(counter),
                        centroid_img)
            counter += 1

        counter = 0
        for index, row in negatives.iterrows():
            if row["CENTRDS"] is str:
                centroid_img = draw_centroids(np.copy(blank_img),
                                              literal_eval(row["CENTRDS"]))
            else:
                centroid_img = draw_centroids(np.copy(blank_img),
                                              row["CENTRDS"])
            # centroid_img = cv2.resize(centroid_img, (224, 224))
            cv2.imwrite(save_directory + "0_{}.png".format(counter),
                        centroid_img)
            counter += 1

    def compute_feature_vectors(dataframe):
        """Use centroid information to calculate row-by-row features."""

        def avg_distance(centroid_list):
            # If loading from csv, convert from str to list
            if type(centroid_list) is str:
                centroid_list = literal_eval(centroid_list)

            dist_sum = 0
            for i in range(len(centroid_list) - 2):
                c1 = centroid_list[i + 1]
                c2 = centroid_list[i + 2]
                dist_sum += math.sqrt(
                    (c2[0] - c1[0]) ** 2 + (c2[1] - c2[1]) ** 2)
                avg_distance = dist_sum / (len(centroid_list) - 2)

            if dist_sum == 0:
                for i in range(len(centroid_list) - 1):
                    c1 = centroid_list[i]
                    c2 = centroid_list[i + 1]
                    dist_sum += math.sqrt(
                        (c2[0] - c1[0]) ** 2 + (c2[1] - c2[1]) ** 2)
                avg_distance = dist_sum / (len(centroid_list) - 1)

            return avg_distance

        def angle_hist_max(centroid_list):
            if type(centroid_list) is str:
                centroid_list = literal_eval(centroid_list)

            bins = np.array([-180, -165, -150, -135, -120, -105,
                             -90, -75, -60, -45, -30, -15,
                             0,
                             15, 30, 45, 60, 75, 90,
                             105, 120, 135, 150, 165])

            histogram = np.zeros(bins.shape)

            for p1 in range(len(centroid_list)):
                for p2 in range(p1 + 1, len(centroid_list)):
                    del_y = centroid_list[p1][0] - centroid_list[p2][0]
                    del_x = -1 * (centroid_list[p1][1] - centroid_list[p2][1])
                    mag = math.sqrt(del_x**2 + del_y**2)
                    angle = math.degrees(math.atan2(del_y, del_x))

                    shifted_angle = angle + 180

                    scaled_angle = shifted_angle / 15
                    high_index = math.ceil(scaled_angle)
                    if high_index == 24:
                        high_index = 0
                    low_index = high_index - 1

                    high_ratio = scaled_angle - low_index
                    low_ratio = 1 - high_ratio

                    histogram[high_index] = high_ratio * mag
                    histogram[low_index] = low_ratio * mag

                    index_max = np.argmax(histogram)
                    angle = bins[index_max]

                    return angle

        def full_angle(centroid_list):
            # If loading from csv, convert from str to list
            if type(centroid_list) is str:
                centroid_list = literal_eval(centroid_list)

            del_y = centroid_list[0][0] - centroid_list[-1][0]
            del_x = -1 * (centroid_list[0][1] - centroid_list[-1][1])
            angle = math.degrees(math.atan2(del_y, del_x))

            return angle

        df_features = pd.DataFrame(index=dataframe.index)

        # AVGDIST didn't distinguish any more than just the ANGLE_F feature
        # df_features["AVGDIST"] = dataframe.apply(
        #     lambda row: avg_distance(row["CENTRDS"]),
        #     axis=1
        # )

        # Flight paths too erratic to consider every angle in path
        # df_features["ANGLE_H"] = dataframe.apply(
        #     lambda row: angle_hist_max(row["CENTRDS"]),
        #     axis=1
        # )

        df_features["ANGLE_F"] = dataframe.apply(
            lambda row: full_angle(row["CENTRDS"]),
            axis=1
        )

        return df_features

    def plot_column_pair(tp_col, tn_col, name):
        save_directory = args.default_dir + args.custom_dir+"feature-testing/"
        if not os.path.isdir(save_directory):
            try:
                os.makedirs(save_directory)
            except OSError:
                print("[!] Creation of the directory {0} failed."
                      .format(save_directory))

        plt.cla()

        ax = tp_col.hist(bins=72, alpha=0.8)
        ax = tn_col.hist(bins=72, alpha=0.5)

        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

        ax.legend(["'Into Chimney' Samples", "'Not Into Chimney' Samples"],
                  loc="upper right")
        ax.set_title("Comparison in {} for Detected Segments".format(name))
        ax.set_xlabel("Angle (Degrees)")
        ax.set_ylabel("Total Segments")
        fig = ax.get_figure()
        fig.savefig(save_directory+'{}.png'.format(name))

    test = None

    # tp, tn = split_data()
    # visualize_path(tp, tn)
    #
    # tp_features = compute_feature_vectors(tp)
    # tn_features = compute_feature_vectors(tn)
    # for column in tp_features.columns:
    #     plot_column_pair(tp_features[column], tn_features[column], column)