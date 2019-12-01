import utils.video_io as vio
import utils.algorithm_testing as test
from swiftwatcher import data_analysis as data

# Parse input arguments
filepath, start, end = vio.parse_filepath_and_framerange()

# Validate input arguments
vio.validate_filepath(filepath)
vio.validate_video_extension(filepath)
vio.validate_framerange(filepath.parent/filepath.stem/"frames",
                        start, end)

# Automatically generate directory for test based on naming scheme
test_dir = test.generate_test_dir(filepath.parent/filepath.stem/"tests")

# Apply swift counting algorithm
config = vio.config_from_file(filepath.parent/filepath.stem/"frames")
events = test.swift_counting_algorithm_from_files(config, start, end)

# Export raw event data to csv
test.dataframe_to_csv(test_dir, events)

# Apply data analysis functions
if not events.empty:
    features = data.generate_feature_vectors(events)
    labels = data.generate_classifications(features)
    total = data.export_results(test_dir, config, labels)

exit(0)
