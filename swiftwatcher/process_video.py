import os
import cv2
import numpy as np


def ms_to_timestamp(total_ms):
    """Helper function to convert millisecond value into formatted timestamp."""
    # cv2's VideoCapture class provides the position of the video in milliseconds
    # (cv2.CAP_PROP_POS_MSEC). However, as it is easier to think of video in terms of
    # hours, minutes, and seconds, it's helpful to convert to a formatted timestamp.
    ms = int(total_ms % 1000)
    s = int((total_ms / 1000) % 60)
    m = int((total_ms / (1000 * 60)) % 60)
    h = int((total_ms / (1000 * 60 * 60)) % 24)
    timestamp = "{0:02d}:{1:02d}:{2:02d}:{3:03d}".format(h, m, s, ms)

    return timestamp


def frameindex_to_timestamp(frame_amount, fps):
    """Helper function to convert an amount of frames into a formatted timestamp."""
    # cv2's VideoCapture class provides a frame count property (cv2.CAP_PROP_FRAME_COUNT)
    # but not a duration property. However, as it is easier to think of video in terms of
    # duration/timestamps, it's helpful to convert back and forth.
    milliseconds = (frame_amount / fps)*1000
    timestamp = ms_to_timestamp(milliseconds)

    return timestamp


def timestamp_to_frameindex(timestamp, fps):
    """Helper function to convert formatted timestamp into an amount of frames."""
    # cv2's VideoCapture class provides a frame count property (cv2.CAP_PROP_FRAME_COUNT)
    # but not a duration property. However, as it is easier to think of video in terms of
    # duration/timestamps, it's helpful to convert back and forth.
    time = timestamp.split(":")
    seconds = (int(time[0])*60*60 +
               int(time[1])*60 +
               int(time[2]) +
               int(time[3])/1000)
    frame_amount = int(seconds * fps)
    return frame_amount


class FrameStack:
    """Class for storing, describing, and manipulating a stack of frames from a video file.

    Attributes: stack, indices, timestamps, src_filename, src_directory, src_fps, src_framecount
    Methods: read_frames, save_frames, save_frames_info, load_frames, load_frames_info
    """
    def __init__(self, video_directory, filename):
        video_filepath = video_directory + filename
        if not os.path.isfile(video_filepath):
            raise Exception("[!] Filepath does not point to valid video file.")

        # Attributes of source file that is associated with FrameStack.
        self.src_filename = filename
        self.src_directory = video_directory
        stream = cv2.VideoCapture("{}/{}".format(self.src_directory, self.src_filename))
        if not stream.isOpened():
            raise Exception("[!] Video file could not be opened to read frames. Check file path.")
        self.src_fps = stream.get(cv2.CAP_PROP_FPS)
        self.src_framecount = stream.get(cv2.CAP_PROP_FRAME_COUNT)
        stream.release()

        # Attributes of FrameStack
        self.stack = []
        self.indices = []
        self.timestamps = []
        self.start_index = None
        self.end_index = None

    def batch_config(self, start_timestamp, end_timestamp, batch_size):
        """Divides requested duration into batches, returns list of timestamp tuples."""
        # Convert provided timestamps into frame indices, calculate total frames in duration
        start_index = timestamp_to_frameindex(start_timestamp, self.src_fps)
        end_index = timestamp_to_frameindex(end_timestamp, self.src_fps)
        frame_count = int(end_index - start_index)

        batch_timestamps = []
        while int(end_index - start_index) > batch_size:
            # Calculate batch timestamps
            batch_start_timestamp = frameindex_to_timestamp(start_index, self.src_fps)
            batch_end_timestamp = frameindex_to_timestamp(start_index+batch_size-1, self.src_fps)
            batch_timestamps.append((batch_start_timestamp, batch_end_timestamp))
            
            # Update timestamp pointer
            start_index += batch_size
        # Append last small batch
        batch_start_timestamp = frameindex_to_timestamp(start_index, self.src_fps)
        batch_end_timestamp = frameindex_to_timestamp(end_index, self.src_fps)
        batch_timestamps.append((batch_start_timestamp, batch_end_timestamp))

        return batch_timestamps

    def read_frames(self, start_timestamp='<Start of File>', end_timestamp='<End of File>',
                    desired_fps='def', verbose=False):
        """Returns a set of frames from a provided input video."""
        # Attempt to open video capture object
        stream = cv2.VideoCapture("{}/{}".format(self.src_directory, self.src_filename))
        if not stream.isOpened():
            raise Exception("[!] Video file could not be opened to read frames. Check file path.")
        
        # Set start index using either provided value or end-of-file.
        if start_timestamp is 'def':
            self.start_index = 0
        else:
            self.start_index = timestamp_to_frameindex(start_timestamp, self.src_fps)
            if not 0 < self.start_index < self.src_framecount:
                raise Exception("[!] Invalid start timestamp. Outside range of acceptable times.")

        # Set end index using either provided value or end-of-file.
        if end_timestamp is 'def':
            self.end_index = self.src_framecount-1
        else:
            self.end_index = timestamp_to_frameindex(end_timestamp, self.src_fps)
            if not self.start_index < self.end_index < self.src_framecount:
                raise Exception("[!] Invalid end timestamp. Outside range of acceptable times.")

        # Set desired fps if not provided.
        if desired_fps is 'def':
            desired_fps = self.src_fps

        # Flush any previously read frame information
        self.stack = []
        self.indices = []
        self.timestamps = []

        # Initialize frame counter and frame index.
        frame_index = self.start_index
        stream.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        frame_count = 0

        print("[========================================================]")
        print("[*] Extracting frames from {} to {}.".format(start_timestamp, end_timestamp))
        while stream.isOpened():
            # Fetch frame
            success, frame = stream.read()

            if not success:
                print("[*] No frame left to be read. Exiting read_frames().")
                stream.release()
                break

            # Get attributes of frame.
            self.stack.append(frame)
            self.indices.append(frame_index)
            self.timestamps.append(ms_to_timestamp(stream.get(cv2.CAP_PROP_POS_MSEC)))

            # Increment frame counter and update frame index.
            frame_count += 1
            if desired_fps == self.src_fps:
                frame_index += 1
            elif 0 < desired_fps < self.src_fps:
                frame_index += round(self.src_fps/desired_fps)
                stream.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            else:
                raise Exception("[!] Invalid FPS. Must be positive and not greater than source file FPS.")
            
            # Provide status updates
            if verbose and frame_count % 100 == 0:
                print("[-] Progress update: {} frames processed.".format(frame_count))

            # Check end condition
            if stream.get(cv2.CAP_PROP_POS_FRAMES) > timestamp_to_frameindex(end_timestamp, self.src_fps):
                print("[-] Specified ending time reached. Exiting read_frames().")
                stream.release()
                break

        print("[-] Frame extraction completed. {} frames extracted.".format(frame_count))

    def save_frames(self, save_directory):
        # TODO: Write a proper docstring for the save_frames() method.
        """Saves a set of frames to a """
        # Check for directory existance
        if not os.path.isdir(save_directory):
            print("[*] Directory %s does not exist. Attempting to create." % save_directory)
            try:
                os.mkdir(save_directory)
            except OSError:
                print("[!] Creation of the directory %s failed." % save_directory)
            else:
                print("[-] Successfully created the directory %s." % save_directory)

        # Save frames to specified directory
        print("[*] Saving frames...")
        count = 0
        for index in range(len(self.stack)):
            try:
                cv2.imwrite("{0}/frame{1}_{2}.png".format(save_directory,
                                                          self.indices[index],
                                                          self.timestamps[index]),
                            self.stack[index])
            except Exception as e:
                print("[!] Image saving failed due to: {0}".format(str(e)))
            count += 1

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

    def convert_grayscale(self):
        print("[*] Converting frames to grayscale...")
        try:
            self.stack = np.array([cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                                   for color_image in self.stack])
            print("[-] Frame conversion successfully completed.")
        except Exception as e:
            print("[!] Frame conversion failed due to: {0}".format(str(e)))

    def crop_frames_rect(self, corners):
        height = corners[1][1] - corners[0][1]
        width = corners[1][0] - corners [0][0]
        print("[*] Cropping frames to size {}x{}...".format(height, width))
        try:
            self.stack = np.array([image[corners[0][1]:corners[1][1], corners[0][0]:corners[1][0]]
                                  for image in self.stack])
            print("[-] Frame cropping successfully completed.")
        except Exception as e:
            print("[!] Frame cropping failed due to: {0}".format(str(e)))






