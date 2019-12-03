import swiftwatcher_refactor.image_processing.data_structures as ds
import swiftwatcher_refactor.io.video_io as vio
import swiftwatcher_refactor.data_analysis.segment_tracking as st


def swift_counting_algorithm(filepath, crop_region, resize_dim, roi_mask):
    """"""

    print("[*] Now processing {}.".format(filepath.name))

    # TODO: Replace FrameReader with VideoCapture object (possibly subclass?)

    # reader = vio.FrameReader(frame_path, start, end)
    queue = ds.FrameQueue()
    tracker = st.SegmentTracker(roi_mask)

    while queue.frames_processed:  # < reader.total_frames:
        # frames, frame_numbers = reader.get_n_frames(n=queue.maxlen)
        queue.fill_queue()  # (frames, frame_numbers)

        queue.preprocess_queue(crop_region, resize_dim)
        queue.segment_queue()

        while not queue.is_empty():
            popped_frame = queue.pop_frame()
            tracker.set_current_frame(popped_frame)

            # tracker.extract_segment_properties()
            # tracker.formulate_cost_matrix()
            # tracker.apply_hungarian_algorithm()
            # tracker.store_matches()
            # tracker.check_for_events()

            tracker.cache_current_frame()

    return tracker.detected_events