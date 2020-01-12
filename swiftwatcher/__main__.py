import swiftwatcher.interface.ui as ui
import swiftwatcher.interface.video_io as vio
import swiftwatcher.interface.data_io as dio
import swiftwatcher.image_processing.image_filtering as img
import swiftwatcher.image_processing.primary_algorithm as alg
import swiftwatcher.data_analysis.event_classification as ec


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
        if events:
            df_events = ec.convert_events_to_dataframe(events,
                                                       ["parent_frame_number",
                                                        "parent_timestamp",
                                                        "centroid"])
            df_labels = ec.classify_events(df_events)

            # Save results to output directory
            dio.export_results(parent_dir, df_labels, properties["fps"],
                               properties["start"], properties["end"])
        else:
            print("[!] No events detected in video '{}'."
                  .format(video_filepath.stem))

if __name__ == "__main__":
    main()
