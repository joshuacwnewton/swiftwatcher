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


def export_results(df_labels):
    test = None