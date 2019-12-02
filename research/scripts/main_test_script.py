"""
    Equivalent to __main__.py in swiftwatcher, but provides additional
    functionality not found in base project. See experimentation.py
"""

import swiftwatcher_refactor.io.ui as ui
import swiftwatcher_refactor.io.video_io as vio
import swiftwatcher_refactor.io.data_io as dio
import swiftwatcher_refactor.image_processing.image_filtering as img
import swiftwatcher_refactor.data_analysis.event_classification as ec
import research.experimentation as exp

# Generate necessary input parameters for swift counting algorithm
filepath, start, end = ui.parse_filepath_and_framerange()
frame_dir = filepath.parent/filepath.stem/"frames"
vio.validate_video_filepath(filepath)
vio.validate_frame_range(frame_dir, start, end)
corners = exp.get_corners_from_file(frame_dir)
crop_region, roi_mask, resize_dim = img.generate_regions(filepath, corners)

# Apply algorithm to detect events and then classify them
events = exp.swift_counting_algorithm(frame_dir,
                                      crop_region, resize_dim, roi_mask,
                                      start, end)
label_dataframes = ec.classify_events(events)

# Save results to unique test directory
test_dir = exp.generate_test_dir(filepath.parent/filepath.stem/"tests")
dio.dataframe_to_csv(test_dir, events)
for label_dataframe in label_dataframes:
    dio.dataframe_to_csv(test_dir, label_dataframe)

exit(0)
