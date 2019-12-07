"""
    Contains functionality for I/O of tabular data. (e.g. pandas
    DataFrames, csv-formatted data, etc.)
"""

from pathlib import Path


def dataframe_to_csv(dataframe, output_filepath):
    if not output_filepath.parent.exists():
        Path.mkdir(output_filepath.parent, parents=True)

    dataframe.to_csv(str(output_filepath))