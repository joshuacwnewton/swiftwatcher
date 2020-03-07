import swiftwatcher.interface.ui as ui
import swiftwatcher.interface.video_io as vio
import swiftwatcher.interface.data_io as dio
import swiftwatcher.image_processing.primary_algorithm as alg
import swiftwatcher.data_analysis.event_classification as ec


def main():
    # 0. Load frame source filepaths and optional debugging arguments
    args = ui.parse_args()
    if len(args.filepaths) > 0:
        src_filepaths = args.filepaths
    else:
        src_filepaths = ui.select_filepaths()

    for src_filepath in src_filepaths:
        # 1. Load frame source into FrameReader object
        if src_filepath.suffix in ['.h5', '.hdf5']:
            reader = vio.HDF5Reader(src_filepath, args.start, args.end)
        else:
            reader = vio.VideoReader(src_filepath, args.end)

        # 2. Specify in-frame corner coordinates
        output_dir = src_filepath.parent / src_filepath.stem
        if (output_dir / "attributes.json").is_file():
            corners = ui.get_corners_from_file(output_dir / "attributes.json")
        else:
            corners = ui.select_chimney_corners(src_filepath)

        # 3. Detect motion which could indicate swifts entering the chimney
        ui.start_status(src_filepath.name)
        events = alg.swift_counting_algorithm(reader, corners, args)

        # 4. If relevant motion was detected, classify instances and export
        if events:
            df_events = ec.convert_events_to_dataframe(events,
                                                       ["parent_frame_number",
                                                        "parent_timestamp",
                                                        "centroid"])
            df_labels = ec.classify_events(df_events)

            if args.debug:
                output_dir = dio.generate_test_dir(output_dir)
            dio.export_results(output_dir, df_labels, reader.fps,
                               reader.start_frame, reader.end_frame)
        else:
            print("[!] No events detected in video '{}'."
                  .format(src_filepath.stem))


if __name__ == "__main__":
    main()
