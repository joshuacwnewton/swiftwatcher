import swiftwatcher_refactor.io.ui as ui
import swiftwatcher_refactor.io.data_io as dio
import swiftwatcher_refactor.io.video_io as vio
import swiftwatcher_refactor.image_processing.composite_algorithms as alg
import swiftwatcher_refactor.data_analysis.event_classification as ec


video_filepaths = ui.select_video_files()
video_attr_list = vio.get_video_attributes(video_filepaths)

for video_attr in video_attr_list:
    events = alg.swift_counting_algorithm(filepath=video_attr["filepath"],
                                          corners=video_attr["corners"])
    label_dataframes = ec.classify_events(events)
    for label_dataframe in label_dataframes:
        dio.dataframe_to_csv(label_dataframe)

