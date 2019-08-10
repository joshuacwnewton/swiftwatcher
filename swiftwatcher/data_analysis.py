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
        test = None

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
