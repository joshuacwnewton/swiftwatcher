import os
import cv2


class FrameStack:
    """Class object for storing, describing, and manipulating a stack of frames.

    Attributes: video_filepath, stack, indices, timestamps
    Methods: read_frames
    """
    def __init__(self, video_filepath):
        if not os.path.isfile(video_filepath):
            raise Exception("Filepath does not point to valid video file.")

        self.source_filepath = video_filepath
        self.source_fps = None
        self.stack = []
        self.indices = []  # As frame storage may skip frames in video, storing absolute indices from video file
        self.timestamps = []

    def read_frames(self, start_frame=0, end_frame='eof', sequence_type='reg', sample_rate='inpt', debug=True):
        """Returns a set of frames from a provided input video.

        Input arguments:
            -start_frame:   Index of frame at the start of the range of frames to read. ('0' = start of file)
            -end_frame:     Index of frame at the end of the range of frames to read. ('default' = end of file)
            -sequence_type: The pattern for how read frames are spaced. ('reg' = regularly spaced)
            -sample_rate:   The rate at which frames are saved from the video file. ('default' = input video frame-rate.)

        Output arguments:
            -frames:     ndarray of frames read from input video file.
            -indices:    List of indices (from original video file) corresponding to each frame.
            -timestamps: List of timestamps (from original video file) corresponding to each frame.
        """

        print("Starting extraction of frames from video file.")
        stream = cv2.VideoCapture(self.source_filepath)
        if not stream.isOpened():
            raise Exception("Video file could not be opened to read frames.")

        # Get attributes of video file
        self.source_fps = stream.get(cv2.CAP_PROP_FPS)

        while True:
            # Fetch frame, check for success
            (grabbed, frame) = stream.read()
            if not grabbed:
                print("Attempted frame not grabbed. Exiting read_frames().")
                break

            # Get attributes of frame
            self.stack.append(frame)
            self.indices.append(int(stream.get(cv2.CAP_PROP_POS_FRAMES))-1)
            self.timestamps.append(stream.get(cv2.CAP_PROP_POS_MSEC))

            # Capping at 10s of video to prevent processing of 1hr files during debug
            if debug:
                if self.timestamps[-1] > 10000:
                    print("10s of frames extracted. Exiting read_frames().")
                    break

    def save_frames(self, frame_folder):
        if os.path.isdir(frame_folder) is False:
            print("Directory %s does not exist. Attempting to create." % frame_folder)
            try:
                os.mkdir(frame_folder)
            except OSError:
                print("Creation of the directory %s failed." % frame_folder)
            else:
                print("Successfully created the directory %s." % frame_folder)

        count = 0
        for index in self.indices:
            try:
                cv2.imwrite(frame_folder + "/test {}.png".format(index), self.stack[index])
            except Exception as err:
                print("Image saving failed due to: {0}".format(str(err)))
            count += 1
        print("{} frames saved.".format(count))
