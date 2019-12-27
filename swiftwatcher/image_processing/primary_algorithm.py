import swiftwatcher.image_processing.data_structures as ds
import swiftwatcher.interface.video_io as vio
import swiftwatcher.data_analysis.segment_tracking as st
import swiftwatcher.interface.ui as ui
import copy


def swift_counting_algorithm(path, crop_region, resize_dim, roi_mask,
                             fps=None, start=0, end=-1, testing=False):
    """Apply individual stages of the multi-stage swift counting
    algorithm to detect potential occurrences of swifts entering
    chimneys."""

    ui.start_status(path.name)

    # Experiments will use subsections of the video (denoted by start/end)
    # read from image files, rather than using the entire video file.
    if testing:
        reader = vio.FrameReader(path, fps, start, end)
    else:
        reader = vio.VideoReader(path)

    queue = ds.FrameQueue()
    tracker = st.SegmentTracker(roi_mask)

    while queue.frames_processed < reader.total_frames:
        # Push frames into queue until full
        frames, frame_numbers, timestamps = reader.get_n_frames(n=queue.maxlen)
        queue.push_list_of_frames(frames, frame_numbers, timestamps)

        # Process an entire queue at once
        queue.preprocess_queue(crop_region, resize_dim)
        queue.segment_queue()  # CPU processing bottleneck

        # Pop frames off queue one-by-one and analyse each separately
        while not queue.is_empty():
            tracker.set_current_frame(queue.pop_frame())
            cost_matrix = tracker.formulate_cost_matrix()
            tracker.store_assignments(st.apply_hungarian_algorithm(cost_matrix))
            tracker.link_matching_segments()
            tracker.check_for_events()
            tracker.cache_current_frame()

        ui.frames_processed_status(queue.frames_processed, reader.total_frames)

    if hasattr(reader, "release"):
        reader.release()

    return copy.deepcopy(tracker.detected_events)
