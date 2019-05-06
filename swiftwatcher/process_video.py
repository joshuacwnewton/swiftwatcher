import os
import cv2


class FrameStack:
    """Class object for storing, describing, and manipulating a stack of frames.

    Attributes: stack, indices, timestamps, src_filepath, src_fps, src_framecount
    Methods: read_frames, save_frames, save_frames_info, load_frames
    """
    def __init__(self, video_filepath):
        if not os.path.isfile(video_filepath):
            raise Exception("[*] Filepath does not point to valid video file.")

        # FrameStack attributes
        self.stack = []
        self.indices = []  # As frame storage may skip frames in video, storing absolute indices from video file
        self.timestamps = []

        # Source file attributes
        self.src_filepath = video_filepath
        self.src_fps = None
        self.src_framecount = None

    def read_frames(self, start_frame=0, end_frame='eof', sequence_type='reg',
                    desired_fps='inpt', verbose=True, debug=True):
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
        stream = cv2.VideoCapture(self.src_filepath)
        if not stream.isOpened():
            raise Exception("Video file could not be opened to read frames.")
        else:
            success = True

        # Get attributes of video file
        self.src_fps = stream.get(cv2.CAP_PROP_FPS)
        self.src_framecount = stream.get(cv2.CAP_PROP_FRAME_COUNT) # May not be valid due to night vision

        n = round(self.src_fps/desired_fps)  # Capture only every nth frame

        while success:
            # Capture only every nth frame (as determined by FPS calculation)
            success = stream.grab()
            current_frame_index = int(stream.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if current_frame_index % n == 0:

                # Fetch frame
                success, frame = stream.retrieve()

                # Get attributes of frame
                # TODO: Format timestamps to be more informative/easier to parse
                self.stack.append(frame)
                self.indices.append(current_frame_index)
                self.timestamps.append(stream.get(cv2.CAP_PROP_POS_MSEC))  # May not be valid due to night vision

            # Status updates for long files
            if verbose:
                if current_frame_index % 1000 == 0:
                    print("[-] {} frames processed.".format(int(stream.get(cv2.CAP_PROP_POS_FRAMES)) - 1))

            if debug:
                # Capping at 10s of video to prevent processing of 1hr files during debug
                if self.timestamps[-1] > 10000:
                    print("[*] 10s of frames extracted. Exiting read_frames().")
                    stream.release()
                    break

        print("[*] Entire video processed. Exiting read_frames().")

    def save_frames(self, save_directory):
        # TODO: Write a proper docstring for the save_frames() method.
        """Saves a set of frames to a """
        # Check for directory existance
        if not os.path.isdir(save_directory):
            print("[-] Directory %s does not exist. Attempting to create." % save_directory)
            try:
                os.mkdir(save_directory)
            except OSError:
                print("[*] Creation of the directory %s failed." % save_directory)
            else:
                print("[-] Successfully created the directory %s." % save_directory)

        # Save frames to specified directory
        # TODO: Rename filenames to accurately reflect the read config
        count = 0
        for index in range(len(self.stack)):
            try:
                cv2.imwrite(save_directory + "/test {}.png".format(index), self.stack[index])
                count += 1
            except Exception as err:
                print("[*] Image saving failed due to: {0}".format(str(err)))

        print("[-] {} frames successfully saved.".format(count))

    def save_frames_info(self, save_directory):
        # TODO: Rename filenames to accurately reflect the read config
        # Save corresponding indices to specified directory
        with open(save_directory + '/indices.txt', 'w') as filehandle:
            filehandle.writelines("%s\n" % index for index in self.indices)

        # Save corresponding timestamps to specified directory
        with open(save_directory + '/timestamps.txt', 'w') as filehandle:
            filehandle.writelines("%s\n" % timestamp for timestamp in self.timestamps)

        print("[-] File information successfully saved.")

    def load_frames(self, save_directory):
        # TODO: Write a proper docstring for the load_frames() method.
        for filename in os.listdir(save_directory):
            frame = cv2.imread(save_directory + '/' + filename)
            self.stack.append(frame)

    def load_frames_info(self, save_directory):
        # TODO: Write a proper docstring for the load_frames() method.
        with open(save_directory + '/indices.txt', 'r') as filehandle:
            self.indices = [int(current_index.rstrip()) for current_index in filehandle.readlines()]

        with open(save_directory + '/timestamps.txt', 'r') as filehandle:
            self.timestamps = [float(current_timestamp.rstrip()) for current_timestamp in filehandle.readlines()]





