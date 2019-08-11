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

import sys
eps = sys.float_info.epsilon


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
        hist, bin_edges = np.histogram(df_features["ANGLE"], 36)

        # mode for continuous variables: https://www.mathstips.com/mode/
        i_max = np.argmax(hist)
        xl = bin_edges[i_max]
        f0 = hist[i_max]
        f_1 = hist[i_max - 1]
        f1 = hist[i_max + 1]
        w = abs(bin_edges[1] - bin_edges[0])
        mode = xl + ((f0 - f_1)/(2*f0 - f1 - f_1))*w
        left = mode - 45
        right = mode + 45

        df_labels = df_features.copy()
        df_labels["ENTERPR"] = np.array([0, 1, 0])[pd.cut(df_features["ANGLE"],
                                                   # bins=[-180, -135, -55, 180]
                                                   bins=[-180 - eps,
                                                         left, right,
                                                         180 + eps],
                                                   labels=False)]

        # Correct errors from 3x3 opened non-birds
        df_labels.loc[(df_labels["ANGLE"] % 15 == 0), "ENTERPR"] = 0

        df_labels["EVENTS"] = 1
    else:
        df_labels = df_features

    return df_labels


def export_results(config, df_labels):
    def create_empty_dataframe():
        # Create Series of DateTimeIndex indices (i.e. frame timestamps)
        frame_queue = FrameQueue(config)
        nano = (1 / frame_queue.src_fps) * 1e9
        frame_queue.stream.release()  # Not needed once fps is extracted
        num_timestamps = frame_queue.total_frames
        duration = (num_timestamps - 1) * nano
        timestamps = pd.date_range(start=config["timestamp"],
                                   end=(pd.Timestamp(config["timestamp"]) +
                                        pd.Timedelta(duration, 'ns')),
                                   periods=num_timestamps)
        timestamps = timestamps.round('us')

        # Create a Series of frame numbers which correspond to the timestamps
        framenumbers = np.array(range(num_timestamps))

        tuples = list(zip(timestamps, framenumbers))
        index = pd.MultiIndex.from_tuples(tuples,
                                          names=['TMSTAMP', 'FRM_NUM'])

        # Create an empty DataFrame for ground truth annotations to be put into
        df_empty = pd.DataFrame(index=index)
        df_empty["PREDICTED"] = None
        df_empty["REJECTED"] = None

        return df_empty

    def split_labeled_events():
        # split into >0 and 0 dataframes
        df_rejected = df_labels[df_labels["ENTERPR"] == 0]
        df_predicted = df_labels[df_labels["ENTERPR"] > 0]

        # groupby sum
        df_rejected = df_rejected.reset_index().groupby(['TMSTAMP',
                                                   'FRM_NUM']).sum()
        df_rejected = df_rejected.drop(columns=["ANGLE", "ENTERPR"])
        df_rejected.columns = ["REJECTED"]

        df_predicted = df_predicted.reset_index().groupby(['TMSTAMP',
                                                     'FRM_NUM']).sum()
        df_predicted = df_predicted.drop(columns=["ANGLE", "ENTERPR"])
        df_predicted.columns = ["PREDICTED"]

        return df_predicted, df_rejected

    def fill_and_group(df_empty, df_predicted, df_rejected):
        # fill none vlaues
        df_empty = df_empty.combine_first(df_rejected)
        df_empty = df_empty.combine_first(df_predicted)
        df_empty = df_empty.fillna(0)

        # create dataframes
        df_exact = df_empty.copy(deep=True)
        df_seconds = df_empty.copy(deep=True)
        df_seconds = \
            df_seconds.set_index(df_seconds.index.levels[0].floor('s'))
        df_seconds = df_seconds.groupby(df_seconds.index).sum()
        df_minutes = df_empty.copy(deep=True)
        df_minutes = \
            df_minutes.set_index(df_minutes.index.levels[0].floor('min'))
        df_minutes = df_minutes.groupby(df_minutes.index).sum()
        df_total = int(np.sum(df_exact["PREDICTED"]))

        return df_total, df_minutes, df_seconds, df_exact

    def save_to_csv(total_count, df_minutes, df_seconds, df_exact):
        nonlocal config

        save_directory \
            = config["src_filepath"].parent/config["src_filepath"].stem

        df_exact_short = df_exact[~((df_exact["PREDICTED"] == 0) &
                                    (df_exact["REJECTED"] == 0))]
        df_minutes_short = df_minutes[~((df_minutes["PREDICTED"] == 0) &
                                        (df_minutes["REJECTED"] == 0))]
        df_seconds_short = df_seconds[~((df_seconds["PREDICTED"] == 0) &
                                        (df_seconds["REJECTED"] == 0))]

        df_exact_short.to_csv(fspath(
                save_directory /
                "{}-swifts_timestamps-exact.csv".format(total_count)
            ))
        df_seconds_short.to_csv(fspath(
                save_directory /
                "{}-swifts_timestamps-seconds.csv".format(total_count)
            ))
        df_minutes_short.to_csv(fspath(
                save_directory /
                "{}-swifts_timestamps-minutes.csv".format(total_count)
            ))
        df_exact.to_csv(fspath(
                save_directory /
                "{}-swifts_timestamps-exact_full.csv".format(total_count)
            ))
        df_seconds.to_csv(fspath(
                save_directory /
                "{}-swifts_timestamps-seconds_full.csv".format(total_count)
            ))
        df_minutes.to_csv(fspath(
                save_directory /
                "{}-swifts_timestamps-minutes_full.csv".format(total_count)
            ))

    empty = create_empty_dataframe()
    predicted, rejected = split_labeled_events()
    total, minutes, seconds, exact = fill_and_group(empty, predicted, rejected)
    save_to_csv(total, minutes, seconds, exact)
