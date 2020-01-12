"""
    Equivalent to __main__.py in swiftwatcher, but provides additional
    experimentation functionality not found in base project.
"""

import swiftwatcher.interface.ui as ui
import swiftwatcher.interface.video_io as vio
import swiftwatcher.interface.data_io as dio
import swiftwatcher.image_processing.image_filtering as img
import swiftwatcher.data_analysis.event_classification as ec
import swiftwatcher.image_processing.primary_algorithm as alg


def main():
    # Parse and validate input arguments
    filepath, start, end = ui.parse_filepath_and_framerange()
    output_dir = filepath.parent/filepath.stem
    # frame_dir = output_dir/"frames"
    h5_path = filepath.parent/f"{filepath.stem}.h5"

    test_dir = dio.generate_test_dir(filepath.parent/filepath.stem/"tests")

    vio.validate_video_filepaths(filepath)
    vio.validate_frame_order(start, end)
    vio.validate_frame_h5(h5_path, start)
    vio.validate_frame_h5(h5_path, end)
    # vio.validate_frame_range(frame_dir, start, end)

    # Use input arguments to get video attributes necessary for algorithm
    properties = vio.get_video_properties_h5(h5_path)
    corners = ui.get_corners_from_file(output_dir/"attributes.json")
    crop_region, roi_mask, resize_dim = img.generate_regions(filepath, corners)

    # Apply algorithm to detect events, then classify detected events
    events = alg.swift_counting_algorithm(h5_path,
                                          crop_region, resize_dim, roi_mask,
                                          properties["fps"], start, end,
                                          test_dir=test_dir)

    if events:
        df_events = ec.convert_events_to_dataframe(events,
                                                   ["parent_frame_number",
                                                    "parent_timestamp",
                                                    "centroid"])
        df_labels = ec.classify_events(df_events)

        # Save results to unique test directory
        dio.dataframe_to_csv(df_events, test_dir/"df_events.csv")
        dio.dataframe_to_csv(df_labels, test_dir/"df_labels.csv")
        dio.export_results(test_dir, df_labels, properties["fps"], start, end)
    else:
        print("[!] No events detected in video '{}'.".format(filepath.stem))


if __name__ == "__main__":
    main()