"""
    Equivalent to __main__.py in swiftwatcher, but provides additional
    experimentation functionality not found in base project.
"""

import swiftwatcher_refactor.interface.ui as ui
import swiftwatcher_refactor.interface.video_io as vio
import swiftwatcher_refactor.interface.data_io as dio
import swiftwatcher_refactor.image_processing.image_filtering as img
import swiftwatcher_refactor.data_analysis.event_classification as ec
import swiftwatcher_refactor.image_processing.primary_algorithm as alg


def main():
    # Generate necessary input parameters for swift counting algorithm
    filepath, start, end = ui.parse_filepath_and_framerange()
    vio.validate_video_filepaths(filepath)
    properties = vio.get_video_properties(filepath)
    corners = ui.get_corners_from_file(output_dir/"attributes.json")
    crop_region, roi_mask, resize_dim = img.generate_regions(filepath, corners)

    # Apply algorithm to detect events, then classify detected events
    events = alg.swift_counting_algorithm(frame_dir,
                                          crop_region, resize_dim, roi_mask,
                                          properties["fps"], start, end,
                                          testing=True)
    df_events = ec.convert_events_to_dataframe(events, ["parent_frame_number",
                                                        "parent_timestamp",
                                                        "centroid"])
    df_labels = ec.classify_events(df_events)

    # Save results to unique test directory
    test_dir = dio.generate_test_dir(filepath.parent/filepath.stem/"tests")
    dio.dataframe_to_csv(df_events, test_dir/"df_events.csv")
    dio.dataframe_to_csv(df_labels, test_dir/"df_labels.csv")
    dio.export_results(test_dir, df_labels, properties["fps"], start, end)


if __name__ == "__main__":
    main()