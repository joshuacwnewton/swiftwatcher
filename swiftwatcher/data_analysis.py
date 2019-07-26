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

# Needed for pairwise iteration
from itertools import tee


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


def generate_classifications(df_features):
    """Classify "segment disappeared" events based on associated feature
    vectors.

    Note: currently this is done using a hard-coded values, but
    if time permits I would like to transition to a ML classifier."""

    df_labels = pd.DataFrame(index=df_features.index)

    df_labels["EXT_CHM"] = np.array([0, 1, 0])[pd.cut(df_features["ANGLE"],
                                               bins=[-180, -125, -55, 180],
                                               labels=False)]

    return df_labels


def generate_comparison(df_eventinfo, df_groundtruth):
    """Generate dataframe comparing events in df_eventinfo with frame
    counts in df_groundtruth."""
    def fix_offbyone(df_comparison):
        def pairwise(iterable):
            """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)

        rows_to_drop = []
        for (i1, row1), (i2, row2) in pairwise(df_comparison.iterrows()):
            if pd.isna(row1["EVENTS"]):
                if (i2[1] == i1[1] + 1) and (
                        row2["LABELPR"] >= row2["LABELGT"]):
                    # Replace nan value with 0 for addition below
                    row1["LABELPR"] = 0
                    row1["EVENTS"] = 0

                    # Merge row values into one row, delete inaccurate row
                    new_values = row1.values + row2.values
                    row1["EVENTS"] = new_values[0]
                    row1["LABELGT"] = new_values[1]
                    row1["LABELPR"] = new_values[2]
                    rows_to_drop.append(i2)

            if pd.isna(row2["EVENTS"]):
                if (i1[1] == i2[1] - 1) and (
                        row1["LABELPR"] >= row1["LABELGT"]):
                    # Replace nan value with 0 for addition below
                    row2["LABELPR"] = 0
                    row2["EVENTS"] = 0

                    # Merge row values into one row, delete inaccurate row
                    new_values = row1.values + row2.values
                    row2["EVENTS"] = new_values[0]
                    row2["LABELGT"] = new_values[1]
                    row2["LABELPR"] = new_values[2]
                    rows_to_drop.append(i1)

        df_comparison_rm = df_comparison.drop(index=rows_to_drop)

        df_comparison_rm = df_comparison_rm.fillna(0)

        return df_comparison_rm

    df_groundtruth = df_groundtruth[df_groundtruth["LABELGT"] > 0]
    df_eventinfo_cp = df_eventinfo.copy()
    df_eventinfo_cp = df_eventinfo_cp.reset_index().groupby(['TMSTAMP',
                                                             'FRM_NUM']).sum()
    df_eventinfo_cp["LABELGT"] = None
    df_combined = df_eventinfo_cp.combine_first(df_groundtruth)
    df_combined["LABELGT"] = df_combined["LABELGT"].fillna(0)

    df_combined_fixed = fix_offbyone(df_combined)

    return df_combined_fixed


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


def evaluate_results(args, df_comparison):
    """Save the bird count estimations from image processing to csv files."""

    def split_comparison(comparison):
        event_types = {}

        # A timestamp contains an event labeled 'positive' if the number of
        # predicted birds is nonzero.
        positives = comparison[comparison["LABELPR"] > 0]
        # A timestamp contains a TP event if the ground truth count
        # (corresponding to a predicted count) is also nonzero.
        event_types["tp"] = positives[positives["LABELGT"] > 0]
        # A timestamp contains a FP event if the predicted count is greater
        # than the corresponding groundtruth count.
        event_types["fp"] = positives[positives["LABELPR"] >
                                      positives["LABELGT"]]

        # A timestamp contains an event labeled 'negative' if the number of
        # predicted birds is less than the number of detected events.
        negatives = comparison[comparison["LABELPR"] < comparison["EVENTS"]]
        # A timestamp contains a TN event if the number of events detected
        # is greater than the ground truth count.
        event_types["tn"] = negatives[negatives["EVENTS"] >
                                      negatives["LABELGT"]]
        # A timestamp contains a FN event if predicted count is less than
        # the corresponding groundtruth count.
        event_types["fn"] = negatives[negatives["LABELPR"] <
                                      negatives["LABELGT"]]

        # A timestamp contains a missed detection if the number of detected
        # events is lower than the ground truth count.
        event_types["md"] = comparison[comparison["EVENTS"] <
                                       comparison["LABELGT"]]

        return event_types

    def sum_counts(event_types, full_comparison):
        sums = {}

        sums["te"] = int(np.sum(full_comparison["EVENTS"]))
        sums["gt"] = int(np.sum(full_comparison["LABELGT"]))

        # mds = total ground truth events - total events detected
        sums["md"] = int(np.sum(np.subtract(event_types["md"]["LABELGT"],
                                            event_types["md"]["EVENTS"])))

        # tps = whichever is lowest between predicted and ground truth events
        sums["tp"] = int(np.sum(np.minimum(event_types["tp"]["LABELPR"],
                                           event_types["tp"]["LABELGT"])))

        # fps = any predicted events that were not present in ground truth
        sums["fp"] = int(np.sum(np.subtract(event_types["fp"]["LABELPR"],
                                            event_types["fp"]["LABELGT"])))

        # A timestamp containing a true negative can also simultaneously have
        # a false positive: both meet criteria of events > ground truth count.
        fps_in_tn = event_types["tn"][event_types["tn"]["LABELPR"] >
                                      event_types["tn"]["LABELGT"]]
        # tns = total criteria-meeting events
        #       - false positive events
        sums["tn"] = int((np.sum(np.subtract(event_types["tn"]["EVENTS"],
                                             event_types["tn"]["LABELGT"]))
                          - np.sum(np.subtract(fps_in_tn["LABELPR"],
                                               fps_in_tn["LABELGT"]))))

        # fns = total detected ground truth events
        #       - total predicted events
        sums["fn"] = int((np.sum(np.minimum(event_types["fn"]["EVENTS"],
                                            event_types["fn"]["LABELGT"]))
                         - np.sum(event_types["fn"]["LABELPR"])))

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
        if not os.path.isdir(save_directory):
            try:
                os.makedirs(save_directory)
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

        file = open(save_directory+'results.txt', 'w')
        file.writelines(results)
        file.close()

    event_dict = split_comparison(df_comparison.copy())
    sum_dict = sum_counts(event_dict, df_comparison)
    export_results(args.default_dir+args.custom_dir+"results/", sum_dict)


def plot_result(args, key, df_prediction, df_groundtruth=None, flag=None):
    """Plot comparisons between estimation and ground truth for segments."""
    save_directory = args.default_dir + args.custom_dir + "results/plots/"
    if not os.path.isdir(save_directory):
        try:
            os.makedirs(save_directory)
        except OSError:
            print("[!] Creation of the directory {0} failed."
                  .format(save_directory))

    df_prediction = df_prediction.reset_index("TMSTAMP")
    es_series = df_prediction[key]

    if df_groundtruth is not None:
        df_groundtruth = df_groundtruth.reset_index("TMSTAMP")
        gt_series = df_groundtruth[key]

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
        title = "False Positive Error for {}".format(key)
        xlabel = "Frame Number"
        ylabel = "False Positives"

    elif flag is "false_negatives":
        series_plots.append(false_negatives.cumsum())
        series_plots.append(false_negatives.rolling(50).sum())
        legend = ["Cumulative Sum", "Rolling Counts"]
        title = "False Negative Error for {}".format(key)
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
    plt.savefig(save_directory + '{0}_{1}.png'.format(key, flag),
                bbox_inches='tight')


def feature_engineering(args, df_comparison):
    """Testing function for exploring different features."""

    def split_data():
        detected_events = df_comparison.dropna()
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

    tp, tn = split_data()
    visualize_path(tp, tn)

    tp_features = compute_feature_vectors(tp)
    tn_features = compute_feature_vectors(tn)
    for column in tp_features.columns:
        plot_column_pair(tp_features[column], tn_features[column], column)