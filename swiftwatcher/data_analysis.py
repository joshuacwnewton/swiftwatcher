# Stdlib imports
from os import fspath
import math
from ast import literal_eval

# Data science libraries
import numpy as np
import pandas as pd

# Needed to generate empty timestamps for exporting results
from swiftwatcher.video_processing import FrameQueue
import sys
eps = sys.float_info.epsilon


def generate_feature_vectors(df_eventinfo):
    """Use segment information to generate feature vectors for each event."""

    def compute_angle(centroid_list):
        # If loading from csv, centroid list may be parsed as string -> fix
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
    """Classify "segment disappeared" events based on corresponding feature
    vectors."""

    def compute_mode():
        """Mode for continuous variables. For more information, see:
        https://www.mathstips.com/mode/ """

        hist, bin_edges = np.histogram(df_features["ANGLE"], bins=36,
                                       range=[-180 - eps, 180 + eps])

        # mode for continuous variables: https://www.mathstips.com/mode/
        i_max = np.argmax(hist)
        xl = bin_edges[i_max]

        if -135 < xl < -45:  # Bugfix to ensure mode is not calculated for
                             # impossible angles
            f0 = hist[i_max]
            f_1 = hist[i_max - 1]
            f1 = hist[i_max + 1]
            w = abs(bin_edges[1] - bin_edges[0])
            estimated_mode = xl + ((f0 - f_1)/(2*f0 - f1 - f_1))*w
        else:
            estimated_mode = -90

        return estimated_mode

    mode = compute_mode()

    df_labels = df_features.copy()
    df_labels["ENTERPR"] = np.array([0, 1, 0])[pd.cut(df_features["ANGLE"],
                                               bins=[-180 - eps,
                                                     mode - 45,
                                                     mode + 45,
                                                     180 + eps],
                                               labels=False)]

    # Correct false positive errors from small (3x3 opened) non-bird segments
    df_labels.loc[(df_labels["ANGLE"] % 15 == 0), "ENTERPR"] = 0

    # Add event count (used for when multiple events occur in a single
    # timestamp -- when rows are combined, "EVENTS" can display as > 1)
    df_labels["EVENTS"] = 1

    return df_labels


def export_results(config, df_labels):
    """Modify event classification dataframe into form that is more suitable
    for output, then save results to csv files."""

    def create_empty_dataframe():
        """Create empty dataframe containing every timestamp in video file."""

        # Create Series of DateTimeIndex indices (i.e. frame timestamps)
        frame_queue = FrameQueue(config)
        nano = (1 / frame_queue.src_fps) * 1e9
        frame_queue.stream.release()  # Not needed once fps is extracted
        num_timestamps = frame_queue.src_framecount
        duration = (num_timestamps - 1) * nano
        timestamps = pd.date_range(start=config["timestamp"],
                                   end=(pd.Timestamp(config["timestamp"]) +
                                        pd.Timedelta(duration, 'ns')),
                                   periods=num_timestamps)
        timestamps = timestamps.round('us')

        # Create a Series of frame numbers which correspond to the timestamps
        framenumbers = np.array(range(num_timestamps))

        # Combine frame numbers and timestamps into multi-index
        tuples = list(zip(timestamps, framenumbers))
        index = pd.MultiIndex.from_tuples(tuples,
                                          names=['TMSTAMP', 'FRM_NUM'])

        # Create an empty DataFrame for ground truth annotations to be put into
        df_empty = pd.DataFrame(index=index)
        df_empty["PREDICTED"] = None
        df_empty["REJECTED"] = None

        return df_empty

    def split_labeled_events():
        """Split event classification dataframes into seperate dataframes for
        predicted and rejected events."""

        # Split dataframe into two different dataframes.
        df_rejected = df_labels[df_labels["ENTERPR"] == 0]
        df_predicted = df_labels[df_labels["ENTERPR"] > 0]

        # Combine multiple events into single rows
        df_rejected = df_rejected.reset_index().groupby(['TMSTAMP',
                                                         'FRM_NUM']).sum()
        df_predicted = df_predicted.reset_index().groupby(['TMSTAMP',
                                                           'FRM_NUM']).sum()

        # Drop unnecessary columns, and rename the remaining "EVENTS" column
        df_rejected = df_rejected.drop(columns=["ANGLE", "ENTERPR"])
        df_predicted = df_predicted.drop(columns=["ANGLE", "ENTERPR"])
        df_rejected.columns = ["REJECTED"]
        df_predicted.columns = ["PREDICTED"]

        return df_predicted, df_rejected

    def fill_and_group(df_empty, df_predicted, df_rejected):
        """Fill previously created empty dataframe with timestamps of detected
        events. Then, create dataframes corresponding to different timestamp
        groupings (e.g. per-minute counts, per-second counts, and exact
        microsecond counts.)"""
        # Fill empty values with detected events
        df_empty = df_empty.combine_first(df_rejected)
        df_empty = df_empty.combine_first(df_predicted)
        df_empty = df_empty.fillna(0)

        # Create dataframes for each time period grouping
        df_exact = df_empty.copy(deep=True)

        df_seconds = df_empty.copy(deep=True)
        df_seconds = \
            df_seconds.set_index(df_seconds.index.levels[0].floor('s'))
        df_seconds = df_seconds.groupby(df_seconds.index).sum()

        df_minutes = df_empty.copy(deep=True)
        df_minutes = \
            df_minutes.set_index(df_minutes.index.levels[0].floor('min'))
        df_minutes = df_minutes.groupby(df_minutes.index).sum()

        # Calculate the total number of predicted swifts
        df_total = int(np.sum(df_exact["PREDICTED"]))

        return df_total, df_minutes, df_seconds, df_exact

    def save_to_csv(count, df_minutes, df_seconds, df_exact):
        """Save counts to csv files in a variety of different formats."""
        nonlocal config

        save_directory \
            = config["src_filepath"].parent/config["src_filepath"].stem

        dfs = {
            "full_usec": df_exact,
            "events-only_usec": df_exact[~((df_exact["PREDICTED"] == 0) &
                                           (df_exact["REJECTED"] == 0))],
            "full_sec": df_seconds,
            "events-only_sec": df_seconds[~((df_seconds["PREDICTED"] == 0) &
                                            (df_seconds["REJECTED"] == 0))],
            "full_min": df_minutes,
            "events-only_min": df_minutes[~((df_minutes["PREDICTED"] == 0) &
                                            (df_minutes["REJECTED"] == 0))]
        }

        for df_name, df in dfs.items():
            df.to_csv(fspath(
                    save_directory/"{0}-swifts_{1}.csv".format(count, df_name)
                ))

    print("[*] Saving results to csv files...")
    empty = create_empty_dataframe()
    predicted, rejected = split_labeled_events()
    total, minutes, seconds, exact = fill_and_group(empty, predicted, rejected)
    save_to_csv(total, minutes, seconds, exact)
