import os
import cv2
import numpy as np


def ms_to_timestamp(total_ms):
    """Helper function to convert millisecond value into formatted timestamp."""
    # cv2's VideoCapture class provides the position of the video in milliseconds
    # (cv2.CAP_PROP_POS_MSEC). However, as it is easier to think of video in terms of
    # hours, minutes, and seconds, it's helpful to convert to a formatted timestamp.
    total_s = int(total_ms/1000)
    total_m = int(total_s/60)
    total_h = int(total_m/60)

    ms = round(total_ms % 1000)
    s = round(total_s % 60)
    m = round(total_m % 60)
    h = round(total_h % 24)

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
    seconds = (float(time[0])*60*60 +
               float(time[1])*60 +
               float(time[2]) +
               float(time[3])/1000)
    frame_amount = int(round(seconds * fps))
    return frame_amount


class FrameStack:
    """Class for storing, describing, and manipulating a stack of frames from a video file.

    Attributes: ssrc_filename, src_directory, src_fps, src_framecount
    Methods: read_frames, save_frames, save_frames_info, load_frames, load_frames_info
    """
    def __init__(self, video_directory, filename):
        # Check validity of filepath
        video_filepath = video_directory + filename
        if not os.path.isfile(video_filepath):
            raise Exception("[!] Filepath does not point to valid video file.")

        # Get attributes from source file that is associated with FrameStack.
        self.src_filename = filename
        self.src_directory = video_directory
        stream = cv2.VideoCapture("{}/{}".format(self.src_directory, self.src_filename))
        if not stream.isOpened():
            raise Exception("[!] Video file could not be opened to read frames. Check file path.")
        else:
            self.src_fps = stream.get(cv2.CAP_PROP_FPS)
            self.src_framecount = stream.get(cv2.CAP_PROP_FRAME_COUNT)
        stream.release()

        # Initialize attributes of FrameStack
        self.stack = []
        self.indices = []
        self.timestamps = []
        self.start_index = None
        self.end_index = None

    def batch_config(self, start_timestamp, end_timestamp, batch_size):
        """Divides requested duration into batches, returns list of timestamp tuples.
        Input arguments:
            -start_timestamp: starting point for video duration to be extracted.
            -end_timestamp: ending point for video duration to be extracted.
            -batch_size: divides full duration into small batches of this size."""

        # Convert provided timestamps into frame indices
        start_index = timestamp_to_frameindex(start_timestamp, self.src_fps)
        end_index = timestamp_to_frameindex(end_timestamp, self.src_fps)

        batch_timestamps = []
        while int(end_index - start_index) > batch_size:
            # Calculate and store batch timestamps
            batch_start_timestamp = frameindex_to_timestamp(start_index, self.src_fps)
            batch_end_timestamp = frameindex_to_timestamp(start_index+batch_size-1, self.src_fps)
            batch_timestamps.append((batch_start_timestamp, batch_end_timestamp))
            
            # Update timestamp pointer
            start_index += batch_size

        # Append remaining small batch (total_frames % batch_size, remainder is the small batch)
        # (Also, returns input timestamps if duration is less than one batch.)
        batch_start_timestamp = frameindex_to_timestamp(start_index, self.src_fps)
        batch_end_timestamp = frameindex_to_timestamp(end_index, self.src_fps)
        batch_timestamps.append((batch_start_timestamp, batch_end_timestamp))

        return batch_timestamps

    def read_frames(self, start_timestamp='<Start of File>', end_timestamp='<End of File>',
                    desired_fps='<Source FPS>', verbose=False):
        """Saves a set of frames into the stack attribute of the class object.

        Input arguments:
            -start_timestamp: starting point for video duration to be extracted.
            -end_timestamp: ending point for video duration to be extracted.
            -desired_fps: allows for subsampling at rates lower than the source FPS.
            -verbose: controls status updates to the console.

        Updated attributes:
            -self.stack: list of RGB frames from specified video duration.
            -self.indices: list of absolute indices corresponding to the original video.
            -self.timestamps: timestamps formatted as 'HH:MM:SS:MSS'."""

        # Attempt to open video capture object
        stream = cv2.VideoCapture("{}/{}".format(self.src_directory, self.src_filename))
        if not stream.isOpened():
            raise Exception("[!] Video file could not be opened to read frames. Check file path.")
        
        # Set start index using either provided value or end-of-file.
        if start_timestamp == '<Start of File>':
            self.start_index = 0
        else:
            self.start_index = timestamp_to_frameindex(start_timestamp, self.src_fps)
            if not 0 <= self.start_index < self.src_framecount:
                raise Exception("[!] Invalid start timestamp. Outside range of acceptable times.")

        # Set end index using either provided value or end-of-file.
        if end_timestamp == '<End of File>':
            self.end_index = self.src_framecount-1
        else:
            self.end_index = timestamp_to_frameindex(end_timestamp, self.src_fps)
            if not self.start_index < self.end_index < self.src_framecount:
                raise Exception("[!] Invalid end timestamp. Outside range of acceptable times.")

        # Set desired fps if not provided.
        if desired_fps == '<Source FPS>':
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
                cv2.imwrite("{0}/frame{1}_{2}.jpg".format(save_directory,
                                                          self.indices[index],
                                                          self.timestamps[index]),
                            self.stack[index])
            except Exception as e:
                print("[!] Image saving failed due to: {0}".format(str(e)))
            count += 1

        print("[-] {} frames successfully saved.".format(count))

    def convert_grayscale(self):
        # TODO: Write a proper docstring for the convert_grayscale() method.
        print("[*] Converting frames to grayscale...")
        try:
            self.stack = np.array([cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                                   for color_image in self.stack])
            print("[-] Frame conversion successfully completed.")
        except Exception as e:
            print("[!] Frame conversion failed due to: {0}".format(str(e)))

    def crop_frames_rect(self, corners):
        # TODO: Write a proper docstring for the crop_frames_rect() method.
        height = corners[1][1] - corners[0][1]
        width = corners[1][0] - corners[0][0]
        print("[*] Cropping frames to size {}x{}...".format(height, width))
        try:
            self.stack = np.array([image[corners[0][1]:corners[1][1], corners[0][0]:corners[1][0]]
                                  for image in self.stack])
            print("[-] Frame cropping successfully completed.")
        except Exception as e:
            print("[!] Frame cropping failed due to: {0}".format(str(e)))

    def resize_frames(self, scale_percent):
        s = scale_percent/100
        self.stack = np.array([cv2.resize(frame,
                                          (round(frame.shape[1]*s), round(frame.shape[0]*s)),
                                          interpolation=cv2.INTER_AREA)
                               for frame in self.stack])
