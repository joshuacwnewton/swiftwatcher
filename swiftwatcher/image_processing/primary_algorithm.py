import swiftwatcher.image_processing.data_structures as ds
import swiftwatcher.image_processing.image_filtering as img
import swiftwatcher.data_analysis.segment_tracking as st
import swiftwatcher.data_analysis.segment_classification as sc
import swiftwatcher.interface.ui as ui
import copy


def swift_counting_algorithm(reader, corners, args):
    """Apply individual stages of the multi-stage swift counting
    algorithm to detect potential occurrences of swifts entering
    chimneys."""

    # Use first frame and coordinates to get regions of interest
    ff = reader.read_frame(0, increment=False)
    crop_region, roi_mask, resize_dim = img.generate_regions(ff, corners)

    ds.Frame.src_video = reader.filepath.stem
    queue = ds.FrameQueue()
    tracker = st.SegmentTracker(roi_mask)
    classifier = sc.SegmentClassifier("swiftwatcher/data_analysis/model.pt")

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
