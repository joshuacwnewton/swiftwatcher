"""
    Contains functionality for I/O of tabular data. (e.g. pandas
    DataFrames, csv-formatted data, etc.)
"""

from pathlib import Path

import numpy as np
import pandas as pd


def dataframe_from_csv(filepath):
    dataframe = pd.read_csv(filepath)
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"]).dt.round(freq='us')
    dataframe.set_index(["timestamp", "framenumber"], inplace=True)

    if "centroid" in dataframe:
        dataframe = list_to_float(dataframe, "centroid")

    return dataframe


def list_to_float(dataframe, column):
    def string_to_float(string_list):
        test_result = string_list.replace(" ", "").replace("[", "").replace("]", "").split("),")
        test_2 = [val.strip("()").split(",") for val in test_result]
        test_3 = [[float(val) for val in tup] for tup in test_2]

        return test_3

    dataframe[column] = dataframe.apply(
        lambda row: string_to_float(row[column]),
        axis=1
    )

    return dataframe


def dataframe_to_csv(dataframe, output_filepath):
    if not output_filepath.parent.exists():
        Path.mkdir(output_filepath.parent, parents=True)

    dataframe.to_csv(str(output_filepath))


def export_results(save_directory, df_labels, fps, start, end):
    """Modify event classification dataframe into form that is more suitable
    for output, then save results to csv files."""

    print("[-]     Saving results to csv files...")
    df_empty = create_empty_dataframe(fps, start, end)
    predicted, rejected = split_labeled_events(df_labels)
    total, minutes, seconds, exact = fill_and_group(df_empty, predicted, rejected)
    save_to_csv(save_directory, total, minutes, seconds, exact)

    return total


def create_empty_dataframe(fps, start, end):
    """Create empty dataframe containing every timestamp in video file."""

    # Create Series of DateTimeIndex indices (i.e. frame timestamps)
    nano = (1 / fps) * 1e9
    num_timestamps = end - start + 1
    duration = (num_timestamps - 1) * nano

    start_timestamp = (pd.Timestamp("00:00:00.000000") +
                       pd.Timedelta(start * nano, 'ns'))
    end_timestamp = (start_timestamp + pd.Timedelta(duration, 'ns'))

    timestamps = pd.date_range(start=start_timestamp, end=end_timestamp,
                               periods=num_timestamps)
    timestamps = timestamps.round(freq='us')

    # Create a Series of frame numbers which correspond to the timestamps
    framenumbers = np.array(range(start, end + 1))

    # Combine frame numbers and timestamps into multi-index
    tuples = list(zip(timestamps, framenumbers))
    index = pd.MultiIndex.from_tuples(tuples,
                                      names=['timestamp', 'framenumber'])

    # Create an empty DataFrame for ground truth annotations to be put into
    df_empty = pd.DataFrame(index=index)
    df_empty["predicted"] = None
    df_empty["rejected"] = None

    return df_empty


def split_labeled_events(df_labels):
    """Split event classification dataframes into seperate dataframes for
    predicted and rejected events."""

    # Split dataframe into two different dataframes.
    df_rejected = df_labels[df_labels["label"] == 0]
    df_predicted = df_labels[df_labels["label"] > 0]

    # Combine multiple events into single rows
    df_rejected = df_rejected.reset_index().groupby(['timestamp',
                                                     'framenumber']).sum()
    df_predicted = df_predicted.reset_index().groupby(['timestamp',
                                                       'framenumber']).sum()

    # Drop unnecessary columns, and rename the remaining "EVENTS" column
    df_rejected = df_rejected.drop(columns=["angle", "label"])
    df_predicted = df_predicted.drop(columns=["angle", "label"])
    df_rejected.columns = ["rejected"]
    df_predicted.columns = ["predicted"]

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
    df_total = int(np.sum(df_exact["predicted"]))

    return df_total, df_minutes, df_seconds, df_exact


def save_to_csv(save_directory, count, df_minutes, df_seconds, df_exact):
    """Save counts to csv files in a variety of different formats."""

    dfs = {
        "full_usec": df_exact,
        "events-only_usec": df_exact[~((df_exact["predicted"] == 0) &
                                       (df_exact["rejected"] == 0))],
        "full_sec": df_seconds,
        "events-only_sec": df_seconds[~((df_seconds["predicted"] == 0) &
                                        (df_seconds["rejected"] == 0))],
        "full_min": df_minutes,
        "events-only_min": df_minutes[~((df_minutes["predicted"] == 0) &
                                        (df_minutes["rejected"] == 0))]
    }

    for df_name, df in dfs.items():
        df.to_csv(str(save_directory/"{0}-swifts_{1}.csv"
                                     .format(count, df_name)))
