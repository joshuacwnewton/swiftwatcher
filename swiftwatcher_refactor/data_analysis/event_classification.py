"""
    An event is triggered when a swift object disappears from frame.
    This file contains functionality to classify those events and
    determine whether the swift disappeared into the in-frame chimney.
"""

import math

import numpy as np
import pandas as pd

import sys
EPSILON = sys.float_info.epsilon


def convert_events_to_dataframe(event_list, attributes_to_keep):
    """Converts list of detected events into a Pandas dataframe.

    Rows of the dataframe correspond to individual events, and columns
    correspond to attributes of the linked segments associated with
    each event. Only the segment attributes specified by
    attributes_to_keep are carried over to the output dataframe."""

    list_of_event_dicts = []

    for event in event_list:
        # Convert list of Segment objects into list of dictionaries
        list_of_dicts = [vars(segment) for segment in event]

        # Convert list of dictionaries into dictionary of lists
        dict_of_lists = {key: [d[key] for d in list_of_dicts]
                         for key in list_of_dicts[0].keys()
                         if key in attributes_to_keep}

        # Extract last frame number/timestamp to use as row MultiIndex
        dict_of_lists["framenumber"] = dict_of_lists["parent_frame_number"][-1]
        dict_of_lists["timestamp"] = dict_of_lists["parent_timestamp"][-1]

        list_of_event_dicts.append(dict_of_lists)

    df_events = pd.DataFrame(list_of_event_dicts)
    df_events.set_index(["timestamp", "framenumber"], inplace=True)

    return df_events


def classify_events(df_events):
    """Take detected events and use their features to classify
    whether those events should truly be counted as a swift entering
    the chimney."""

    df_features = generate_angle_features(df_events)
    df_features_filtered = filter_false_angles(df_features)
    df_labels = generate_classifications(df_features_filtered)

    # Add event count (used for when multiple events occur in a single
    # timestamp -- when rows are merged, "events" can display as > 1)
    df_labels["events"] = 1

    return df_labels


def generate_angle_features(df_events):
    """Use segment information to generate feature vectors for each event."""

    df_features = pd.DataFrame(index=df_events.index)
    df_features["angle"] = df_events.apply(
        lambda row: compute_angle(row["centroid"]),
        axis=1
    )

    return df_features


def compute_angle(centroid_list):
    """Compute the angle between the first and last coordinate within
    the list."""

    del_y = centroid_list[0][0] - centroid_list[-1][0]
    del_x = -1 * (centroid_list[0][1] - centroid_list[-1][1])
    angle = math.degrees(math.atan2(del_y, del_x))

    return angle


def filter_false_angles(df_features):
    """Remove angles that are exact multiples of 15 degrees. These
    angles almost never occur naturally due to the size and complexity
    of bird segments.

    This is a cheap solution that could be improved by filtering out
    unnatural segments themselves."""

    # Correct false positive errors from small (3x3 opened) non-bird segments
    index_to_drop = df_features[(df_features["angle"] % 15 == 0)].index
    if not index_to_drop.empty:
        df_features = df_features.drop(
            df_features[(df_features["angle"] % 15 == 0)].index)

    return df_features


def generate_classifications(df_features):
    """Classify "segment disappeared" events based on corresponding feature
    vectors."""

    mode = compute_mode(df_features)

    df_labels = df_features.copy()
    df_labels["label"] = np.array([0, 1, 0])[pd.cut(df_features["angle"],
                                             bins=[-180 - EPSILON,
                                                   mode - 30,
                                                   mode + 30,
                                                   180 + EPSILON],
                                             labels=False)]

    return df_labels


def compute_mode(df_features):
    """Mode for continuous variables. For more information, see:
    https://www.mathstips.com/mode/ """

    hist, bin_edges = np.histogram(df_features["angle"], bins=36,
                                   range=[-180 - EPSILON, 180 + EPSILON])

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
