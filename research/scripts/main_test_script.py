"""
    Equivalent to __main__.py in swiftwatcher, but provides additional
    functionality not found in base project.
"""

import swiftwatcher_refactor.io.ui as ui
import swiftwatcher_refactor.io.video_io as vio
import swiftwatcher_refactor.io.data_io as dio
import swiftwatcher_refactor.image_processing.image_filtering as img
import swiftwatcher_refactor.data_analysis.event_classification as ec
import swiftwatcher_refactor.image_processing.primary_algorithm as alg

# Generate necessary input parameters for swift counting algorithm
filepath, start, end = ui.parse_filepath_and_framerange()
vio.validate_video_filepath(filepath)
properties = vio.get_video_properties(filepath)

frame_dir = filepath.parent/filepath.stem/"frames"
vio.validate_frame_range(frame_dir, start, end)

corners = vio.get_corners_from_file(frame_dir)
crop_region, roi_mask, resize_dim = img.generate_regions(filepath, corners)

# Apply algorithm to detect events, then classify detected events
events = alg.swift_counting_algorithm(frame_dir,
                                      crop_region, resize_dim, roi_mask,
                                      properties["fps"], start, end)
df_events = ec.convert_events_to_dataframe(events, ["parent_frame_number",
                                                    "parent_timestamp",
                                                    "centroid"])
df_labels = ec.classify_events(df_events)

# Save results to unique test directory
test_dir = vio.generate_test_dir(filepath.parent/filepath.stem/"tests")
dio.dataframe_to_csv(df_events, test_dir/"df_events.csv")
dio.dataframe_to_csv(df_labels, test_dir/"df_labels.csv")

# for label_dataframe in label_dataframes:
#     dio.dataframe_to_csv(test_dir, label_dataframe)

exit(0)
