import swiftwatcher_refactor.interface.ui as ui
import swiftwatcher_refactor.interface.video_io as vio
import swiftwatcher_refactor.interface.data_io as dio
import swiftwatcher_refactor.image_processing.image_filtering as img
import swiftwatcher_refactor.image_processing.primary_algorithm as alg
import swiftwatcher_refactor.data_analysis.event_classification as ec


def main():
    video_filepaths = ui.select_filepaths()
    vio.validate_video_filepaths(video_filepaths)

    for video_filepath in video_filepaths:
        # Initialize variables needed to execute the algorithm
        parent_dir = video_filepath.parent / video_filepath.stem
        properties = vio.get_video_properties(video_filepath)
        corners = ui.select_chimney_corners(video_filepath)
        crop_region, roi_mask, resize_dim \
            = img.generate_regions(video_filepath, corners)

        # Detect frames which contain "swifts entering chimney"
        events = alg.swift_counting_algorithm(video_filepath, crop_region,
                                              resize_dim, roi_mask)
        df_events = ec.convert_events_to_dataframe(events,
                                                   ["parent_frame_number",
                                                    "parent_timestamp",
                                                    "centroid"])
        df_labels = ec.classify_events(df_events)

        # Save results to output directory
        dio.export_results(parent_dir, df_labels, properties["fps"],
                           properties["start"], properties["end"])


if __name__ == "__main__":
    main()
