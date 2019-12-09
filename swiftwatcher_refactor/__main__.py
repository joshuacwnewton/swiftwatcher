import swiftwatcher_refactor.io.ui as ui
import swiftwatcher_refactor.io.data_io as dio
import swiftwatcher_refactor.image_processing.image_filtering as img
import swiftwatcher_refactor.image_processing.primary_algorithm as alg
import swiftwatcher_refactor.data_analysis.event_classification as ec


def main():
    video_filepaths = ui.select_video_files()

    for video_filepath in video_filepaths:
        corners = ui.select_chimney_corners(video_filepath)
        crop_region, roi_mask, resize_dim = img.generate_regions(video_filepath,
                                                                 corners)
        events = alg.swift_counting_algorithm(video_filepath,
                                              crop_region, resize_dim, roi_mask)
        df_events = ec.convert_events_to_dataframe(events, ["parent_frame_number",
                                                            "parent_timestamp",
                                                            "centroid"])
        df_labels = ec.classify_events(df_events)

        # Save results to unique test directory
        parent_dir = video_filepath.parent / video_filepath.stem
        dio.dataframe_to_csv(df_events, parent_dir / "df_events.csv")
        dio.dataframe_to_csv(df_labels, parent_dir / "df_labels.csv")

