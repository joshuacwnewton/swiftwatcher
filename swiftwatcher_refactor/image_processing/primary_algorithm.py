from swiftwatcher_refactor.image_processing.data_structures import FrameQueue


def swift_counting_algorithm(filepath, crop_region, resize_dim, roi_mask):
    """"""

    print("[*] Now processing {}.".format(filepath.name))

    fq = FrameQueue(src_path=filepath.parent/filepath.stem/"frames")

    while fq.frames_read < fq.total_frames:
        fq.fill_queue()
        fq.preprocess_queue(crop_region, resize_dim)
        fq.segment_queue()

    return fq