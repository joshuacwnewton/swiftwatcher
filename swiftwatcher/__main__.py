import swiftwatcher.ui as ui
import swiftwatcher.io_video as vio
import swiftwatcher.io_data as dio
import swiftwatcher.data_structures as ds
import swiftwatcher.image_filtering as img
import swiftwatcher.segment_tracking as st
import swiftwatcher.segment_classification as sc
import swiftwatcher.event_classification as ec

import copy


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
        events = swift_counting_algorithm(reader, corners, args)

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


def swift_counting_algorithm(reader, corners, args):
    """Apply individual stages of the multi-stage swift counting
    algorithm to detect potential occurrences of swifts entering
    chimneys."""

    # Use first frame and coordinates to get regions of interest
    ff = reader.read_frame(0, increment=False)
    crop_region, roi_mask, resize_dim = img.generate_regions(ff, corners)

    # Initialize data structures needed for tracking/classification
    ds.Frame.src_video = reader.filepath.stem
    queue = ds.FrameQueue()
    tracker = st.SegmentTracker(roi_mask)
    classifier = sc.SegmentClassifier("swiftwatcher/model.pt")

    while queue.frames_processed < reader.total_frames:
        # Push frames into queue until full
        frames, frame_numbers, timestamps = reader.get_n_frames(n=queue.maxlen)
        queue.push_list_of_frames(frames, frame_numbers, timestamps)

        # Process an entire queue at once
        queue.preprocess_queue(crop_region, resize_dim)
        queue.segment_queue((24, 24), crop_region)  # CPU processing bottleneck

        # Pop frames off queue one-by-one and analyse each separately
        while not queue.is_empty():
            popped_frame = queue.pop_frame()

            if args.classify:
                popped_frame.segments = classifier(popped_frame.segments)

            tracker.set_current_frame(popped_frame)
            cost_matrix = tracker.formulate_cost_matrix()
            tracker.store_assignments(st.apply_hungarian_algorithm(cost_matrix))
            tracker.link_matching_segments()
            tracker.check_for_events()
            tracker.cache_current_frame()

            if args.export:
                popped_frame.export_segments((24, 24), crop_region,
                                             reader.filepath/"segments")

        ui.frames_processed_status(queue.frames_processed, reader.total_frames)

    return copy.deepcopy(tracker.detected_events)


if __name__ == "__main__":
    main()
