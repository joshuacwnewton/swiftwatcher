import utils.video_io as vio
from swiftwatcher import algorithm as alg
from swiftwatcher import data_analysis as data

# Parse input arguments
filepath, start, end = vio.parse_filepath_and_framerange()

# Validate input arguments
vio.validate_filepath(filepath)
vio.validate_video_extension(filepath)
vio.validate_framerange(filepath.parent/filepath.stem/"frames",
                        start, end)

# Apply swift counting algorithm
config = vio.config_from_file(filepath.parent/filepath.stem/"frames")
events = alg.swift_counting_algorithm_from_frames(config, start, end)

# Apply data analysis functions
features = data.generate_feature_vectors(events)
labels = data.generate_classifications(features)
total = data.export_results(config, labels)
