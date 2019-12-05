"""
    An event is triggered when a swift object disappears from frame.
    This file contains functionality to classify those events and
    determine whether the swift disappeared into the in-frame chimney.
"""

import pandas as pd


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


def classify_events(events):
    return None