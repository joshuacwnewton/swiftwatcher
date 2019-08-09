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
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

    def compute_poly_slope(centroid_list):
        if type(centroid_list) is str:
            centroid_list = literal_eval(centroid_list)

        x, y = zip(*centroid_list)
        slope, y_int = np.polyfit(x, y, 1)

        return slope

    def compute_poly_yint(centroid_list):
        if type(centroid_list) is str:
            centroid_list = literal_eval(centroid_list)

        x, y = zip(*centroid_list)
        slope, y_int = np.polyfit(x, y, 1)

        return y_int

    if not df_eventinfo.empty:
        df_features = pd.DataFrame(index=df_eventinfo.index)
        df_features["ANGLE"] = df_eventinfo.apply(
            lambda row: compute_angle(row["CENTRDS"]),
            axis=1
        )
        df_features["SLOPE"] = df_eventinfo.apply(
            lambda row: compute_poly_slope(row["CENTRDS"]),
            axis=1
        )
        df_features["Y_INT"] = df_eventinfo.apply(
            lambda row: compute_poly_yint(row["CENTRDS"]),
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
    def train_model():
        positives = pd.read_csv(fspath(Path.cwd()/"videos"/"positives.csv"))
        negatives = pd.read_csv(
            fspath(Path.cwd() / "videos" / "negatives.csv"))
        X_p = np.array([positives["SLOPE"].values, positives["Y_INT"].values]).T
        X_n = np.array([negatives["SLOPE"].values, negatives["Y_INT"].values]).T
        y_p = np.ones((X_p.shape[0], 1))
        y_n = np.zeros((X_n.shape[0], 1))
        X = np.vstack([X_p, X_n])
        y = np.vstack([y_p, y_n])

        clf = GaussianProcessClassifier(1.0 * RBF(1.0))
        # X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        return clf

    def classify_feature_vectors(clf):
        # Hand-crafted classification
        df_labels["SLLABEL"] = np.array([0, 1, 0])[pd.cut(df_features["SLOPE"],
                                                   bins=[-1e6, -1.5, 1.5, 1e6],
                                                   labels=False)]
        df_labels["YILABEL"] = np.array([0, 1, 0])[pd.cut(df_features["Y_INT"],
                                                   bins=[-1e6, -250, 250, 1e6],
                                                   labels=False)]
        y1 = df_labels["YILABEL"] & df_labels["SLLABEL"]

        # new method using classifier
        classifier = train_model()
        X = np.array([df_labels["SLOPE"].values, df_labels["Y_INT"].values]).T
        # X = StandardScaler().fit_transform(X)
        y2 = classifier.predict(X).T
        return y1, y2

    if not df_features.empty:
        df_labels = df_features.copy()
        df_labels["ENTERPR"] = np.array([0, 1, 0])[pd.cut(df_features["ANGLE"],
                                                   bins=[-180, -125, -55, 180],
                                                   labels=False)]

        # Experimental classification using line-fitting
        # classifier = train_model()
        # df_labels["ENTERPR2"], df_labels["ENTERPR3"] \
        #     = classify_feature_vectors(classifier)

        # Give each classified event a value of 1, so that when multiple events
        # on a single timestamp are merged, it will clearly show EVENTS = (>=2)
        df_labels["EVENTS"] = 1
    else:
        df_labels = df_features

    return df_labels


def generate_comparison(config, df_prediction, df_groundtruth):
    """Generate dataframe comparing events in df_eventinfo with frame
    counts in df_groundtruth."""
    def fix_offbyonetwo(df_comparison):
        df_comparison = df_comparison.fillna(0)

        if type(df_comparison.index) is pd.MultiIndex:
            for offset in [1, 2]:
                rows_to_drop = []
                for (i1, row1) in df_comparison.iterrows():
                    i1 = i1[1]
                    i2 = i1 + offset
                    if i2 in df_comparison.index.get_level_values(1):
                        row2 = df_comparison.xs(i2, level=1, drop_level=False)
                        row2T = row2.T.squeeze()
                        diff1 = row1["ENTERGT"] - row1["ENTERPR"]
                        diff2 = row2T["ENTERGT"] - row2T["ENTERPR"]

                        # Condition for FN/MD and FP in sequential frames
                        if (diff1 > 0) and (diff2 < 0):
                            # Shift "off-by-one" GT count to cancel out errors
                            offbyone = min(diff1, abs(diff2))
                            row1["ENTERGT"] -= offbyone
                            df_comparison.at[row2.index.remove_unused_levels().values[0], "ENTERGT"] += offbyone
                            # Remove row if empty (e.g. 1 0 0 -> 0 0 0 after shift)
                            if np.array_equal(row1.values, [0, 0, 0]):
                                rows_to_drop.append(i1)

                        # Condition for FP and FN/MD in sequential frames
                        elif (diff1 < 0) and (diff2 > 0):
                            # Shift "off-by-one" GT count to cancel out errors
                            offbyone = min(abs(diff1), diff2)
                            df_comparison.at[row2.index.remove_unused_levels().values[0], "ENTERGT"] -= offbyone
                            row1["ENTERGT"] += offbyone

                            # Remove row if empty (e.g. 1 0 0 -> 0 0 0 after shift)
                            if np.array_equal(row2.values, [0, 0, 0]):
                                rows_to_drop.append(i2)
                df_comparison = df_comparison.drop(level=1, index=rows_to_drop)
        else:
            for offset in [1, 2]:
                rows_to_drop = []
                for (i1, row1) in df_comparison.iterrows():
                    i2 = i1 + offset
                    if i2 in df_comparison.index:
                        row2 = df_comparison.loc[i2, :]
                        diff1 = row1["ENTERGT"] - row1["ENTERPR"]
                        diff2 = row2["ENTERGT"] - row2["ENTERPR"]

                        # Condition for FN/MD and FP in sequential frames
                        if (diff1 > 0) and (diff2 < 0):
                            # Shift "off-by-one" GT count to cancel out errors
                            offbyone = min(diff1, abs(diff2))
                            row1["ENTERGT"] -= offbyone
                            df_comparison.at[i2, "ENTERGT"] += offbyone
                            # Remove row if empty (e.g. 1 0 0 -> 0 0 0 after shift)
                            if np.array_equal(row1.values, [0, 0, 0]):
                                rows_to_drop.append(i1)

                        # Condition for FP and FN/MD in sequential frames
                        elif (diff1 < 0) and (diff2 > 0):
                            # Shift "off-by-one" GT count to cancel out errors
                            offbyone = min(abs(diff1), diff2)
                            df_comparison.at[i2, "ENTERGT"] -= offbyone
                            row1["ENTERGT"] += offbyone

                            # Remove row if empty (e.g. 1 0 0 -> 0 0 0 after shift)
                            if np.array_equal(row2.values, [0, 0, 0]):
                                rows_to_drop.append(i2)
                df_comparison = df_comparison.drop(index=rows_to_drop)

        return df_comparison

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
        df_combined_fixed = fix_offbyonetwo(df_combined)
        df_combined_fixed = df_combined_fixed.drop(index=indexes_to_drop)
    else:
        df_combined_fixed = df_prediction

    return df_combined, df_combined_fixed


def import_dataframes(load_directory, df_list):
    dfs = {}
    for df_name in df_list:
        if df_name == "groundtruth":
            dfs[df_name] = \
                pd.read_csv(fspath(load_directory.parent.parent/
                                   "{}.csv".format(df_name)))
        else:
            dfs[df_name] = \
                pd.read_csv(fspath(load_directory / "results" / "df-export" /
                                   "{}.csv".format(df_name)))

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

        event_types["p"] = pd.concat([event_types["tp"], event_types["fn"]])
        event_types["n"] = pd.concat([event_types["fp"], event_types["tn"]])

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


def feature_engineering(args, config, results):
    def plot_features(dataframe_list, feature_list, config_dict, plot_name):
        plt.cla()

        for dataframe in dataframe_list:
            if len(feature_list) == 1:
                feature = feature_list[0]
                x = dataframe[feature].values

                n, bins, patches = plt.hist(x, range=[-180, 180],
                                            density=True,
                                            histtype='stepfilled',
                                            bins=36, alpha=0.5)

                if plot_name is "positives":
                    from scipy.stats import norm
                    mu, std = norm.fit(x)
                    test = None

                ax = plt.gca()
                ax.minorticks_on()
                ax.grid(which='major', linestyle='-', linewidth='0.5',
                        color='black')
                ax.grid(which='minor', linestyle=':', linewidth='0.5',
                        color='black')


            if len(feature_list) == 2:
                feature = feature_list[0].join(feature_list[1])
                plt.scatter(dataframe[feature_list[0]],
                            dataframe[feature_list[1]], s=0.5)


        save_directory = (Path.cwd()/"feature-engineering"
                          /args.custom_dir/config_dict["name"])
        if not os.path.isdir(save_directory):
            try:
                os.makedirs(save_directory)
            except OSError:
                print("[!] Creation of the directory {0} failed."
                      .format(save_directory))
        plt.savefig(fspath(save_directory/"{0}-{1}.png"
                           .format(feature, plot_name)), dpi=300)

    config.append({"name": "NPD June 13 and June 14"})
    results.append({
        "p": pd.concat([results[1]["p"], results[2]["p"]]),
        "n": pd.concat([results[1]["n"], results[2]["n"]]),
    })
    config.append({"name": "ch04 and NPD June 13 and June 14"})
    results.append({
        "p": pd.concat([results[0]["p"], results[1]["p"], results[2]["p"]]),
        "n": pd.concat([results[0]["n"], results[1]["n"], results[2]["n"]])
    })

    counter = 0
    for result_dict in results:
        for key, value in result_dict.items():
            result_dict[key] = \
                result_dict[key][~(result_dict[key]["EVENTS"] > 1)]

        for features in [["ANGLE"], ["SLOPE", "Y_INT"]]:
            plot_features([result_dict["p"]], features, config[counter],
                          "positives")
            plot_features([result_dict["n"]], features, config[counter],
                          "negatives")
            plot_features([result_dict["p"], result_dict["n"]], features,
                          config[counter], "positives-negatives")

        counter += 1