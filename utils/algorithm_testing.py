from datetime import date
from glob import glob
from pathlib import Path


def generate_test_dir(parent_dir):
    """Generate test directory based on the following scheme:
        parent_dir/<today's date>/<ID of last test + 1>

    If no test has been run today, set ID to 1."""

    # Set base testing directory to today's date
    date_dir = parent_dir / str(date.today())

    if not date_dir.exists():
        # Date directory doesnt exist, so must be first test run today
        test_dir = date_dir / "1"

    else:
        # Fetch names of all subdirectories in date_dir, then get the max
        last_test_id = max([int(path.stem) for path in
            [Path(path_str) for path_str in glob(str(date_dir / "*/"))]])

        # Set test directory to last test incremented by one
        test_dir = date_dir / str(last_test_id + 1)

    return test_dir


def dataframe_to_csv(test_dir, input_dataframe):
    if not test_dir.exists():
        Path.mkdir(test_dir, parents=True)

    input_dataframe.to_csv(str(test_dir / "events.csv"))


def csv_to_dataframe():
    """"""

