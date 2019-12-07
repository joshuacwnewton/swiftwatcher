"""
    Contains functionality for video I/O, as well as video frame I/O.
"""

import sys
from pathlib import Path
from glob import glob
import re

import json
from datetime import date

import pandas as pd
import numpy as np
import cv2


###############################################################################
#                        VALIDATION FUNCTIONS BEGIN HERE                      #
###############################################################################


def validate_directory(dirpath):
    """Ensure that directory path points to valid directory """
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        sys.stderr.write("Error: {} doesn't point to directory."
                         .format(dirpath))
        sys.exit()


def validate_video_filepath(video_filepath):
    """Basic checks on a given video filepath"""

    validate_filepath(video_filepath)
    validate_video_read(video_filepath)


def validate_filepath(filepath):
    """Ensure that file path points to a valid file."""

    # Ensure that path is Path object
    filepath = Path(filepath)

    if not Path.is_file(filepath):
        sys.stderr.write("[!] Error: {} does not point to a valid file."
                         .format(filepath.name))
        sys.exit()


def validate_video_read(video_filepath):
    """Ensure that frames can be read from video file."""

    vidcap = cv2.VideoCapture(str(video_filepath))
    retval, _ = vidcap.read()

    if retval is False:
        sys.stderr.write("[!] Error: Unable to read frames from {}."
                         .format(video_filepath.name))
        sys.exit()


def validate_video_extension(video_filepath):
    """Ensure that extension of file is a video file type. May be
    unnecessary with validate_video_read, but keeping to commit at least
    once."""

    video_file_extensions = (
        '.264', '.3g2', '.3gp', '.3gp2', '.3gpp', '.3gpp2', '.3mm', '.3p2',
        '.60d', '.787', '.89', '.aaf', '.aec', '.aep', '.aepx',
        '.aet', '.aetx', '.ajp', '.ale', '.am', '.amc', '.amv', '.amx',
        '.anim', '.aqt', '.arcut', '.arf', '.asf', '.asx', '.avb',
        '.avc', '.avd', '.avi', '.avp', '.avs', '.avs', '.avv', '.axm',
        '.bdm', '.bdmv', '.bdt2', '.bdt3', '.bik', '.bin', '.bix',
        '.bmk', '.bnp', '.box', '.bs4', '.bsf', '.bvr', '.byu', '.camproj',
        '.camrec', '.camv', '.ced', '.cel', '.cine', '.cip',
        '.clpi', '.cmmp', '.cmmtpl', '.cmproj', '.cmrec', '.cpi', '.cst',
        '.cvc', '.cx3', '.d2v', '.d3v', '.dat', '.dav', '.dce',
        '.dck', '.dcr', '.dcr', '.ddat', '.dif', '.dir', '.divx', '.dlx',
        '.dmb', '.dmsd', '.dmsd3d', '.dmsm', '.dmsm3d', '.dmss',
        '.dmx', '.dnc', '.dpa', '.dpg', '.dream', '.dsy', '.dv', '.dv-avi',
        '.dv4', '.dvdmedia', '.dvr', '.dvr-ms', '.dvx', '.dxr',
        '.dzm', '.dzp', '.dzt', '.edl', '.evo', '.eye', '.ezt', '.f4p',
        '.f4v', '.fbr', '.fbr', '.fbz', '.fcp', '.fcproject',
        '.ffd', '.flc', '.flh', '.fli', '.flv', '.flx', '.gfp', '.gl',
        '.gom', '.grasp', '.gts', '.gvi', '.gvp', '.h264', '.hdmov',
        '.hkm', '.ifo', '.imovieproj', '.imovieproject', '.ircp', '.irf',
        '.ism', '.ismc', '.ismv', '.iva', '.ivf', '.ivr', '.ivs',
        '.izz', '.izzy', '.jss', '.jts', '.jtv', '.k3g', '.kmv', '.ktn',
        '.lrec', '.lsf', '.lsx', '.m15', '.m1pg', '.m1v', '.m21',
        '.m21', '.m2a', '.m2p', '.m2t', '.m2ts', '.m2v', '.m4e', '.m4u',
        '.m4v', '.m75', '.mani', '.meta', '.mgv', '.mj2', '.mjp',
        '.mjpg', '.mk3d', '.mkv', '.mmv', '.mnv', '.mob', '.mod', '.modd',
        '.moff', '.moi', '.moov', '.mov', '.movie', '.mp21',
        '.mp21', '.mp2v', '.mp4', '.mp4v', '.mpe', '.mpeg', '.mpeg1',
        '.mpeg4', '.mpf', '.mpg', '.mpg2', '.mpgindex', '.mpl',
        '.mpl', '.mpls', '.mpsub', '.mpv', '.mpv2', '.mqv', '.msdvd',
        '.mse', '.msh', '.mswmm', '.mts', '.mtv', '.mvb', '.mvc',
        '.mvd', '.mve', '.mvex', '.mvp', '.mvp', '.mvy', '.mxf', '.mxv',
        '.mys', '.ncor', '.nsv', '.nut', '.nuv', '.nvc', '.ogm',
        '.ogv', '.ogx', '.osp', '.otrkey', '.pac', '.par', '.pds', '.pgi',
        '.photoshow', '.piv', '.pjs', '.playlist', '.plproj',
        '.pmf', '.pmv', '.pns', '.ppj', '.prel', '.pro', '.prproj',
        '.prtl', '.psb', '.psh', '.pssd', '.pva', '.pvr', '.pxv',
        '.qt', '.qtch', '.qtindex', '.qtl', '.qtm', '.qtz', '.r3d', '.rcd',
        '.rcproject', '.rdb', '.rec', '.rm', '.rmd', '.rmd',
        '.rmp', '.rms', '.rmv', '.rmvb', '.roq', '.rp', '.rsx', '.rts',
        '.rts', '.rum', '.rv', '.rvid', '.rvl', '.sbk', '.sbt',
        '.scc', '.scm', '.scm', '.scn', '.screenflow', '.sec', '.sedprj',
        '.seq', '.sfd', '.sfvidcap', '.siv', '.smi', '.smi',
        '.smil', '.smk', '.sml', '.smv', '.spl', '.sqz', '.srt', '.ssf',
        '.ssm', '.stl', '.str', '.stx', '.svi', '.swf', '.swi',
        '.swt', '.tda3mt', '.tdx', '.thp', '.tivo', '.tix', '.tod', '.tp',
        '.tp0', '.tpd', '.tpr', '.trp', '.ts', '.tsp', '.ttxt',
        '.tvs', '.usf', '.usm', '.vc1', '.vcpf', '.vcr', '.vcv', '.vdo',
        '.vdr', '.vdx', '.veg', '.vem', '.vep', '.vf', '.vft',
        '.vfw', '.vfz', '.vgz', '.vid', '.video', '.viewlet', '.viv',
        '.vivo', '.vlab', '.vob', '.vp3', '.vp6', '.vp7', '.vpj',
        '.vro', '.vs4', '.vse', '.vsp', '.w32', '.wcp', '.webm', '.wlmp',
        '.wm', '.wmd', '.wmmp', '.wmv', '.wmx', '.wot', '.wp3',
        '.wpl', '.wtv', '.wve', '.wvx', '.xej', '.xel', '.xesc', '.xfl',
        '.xlmv', '.xmv', '.xvid', '.y4m', '.yog', '.yuv', '.zeg',
        '.zm1', '.zm2', '.zm3', '.zmv')

    # Ensure that video_filepath is a Path object
    video_filepath = Path(video_filepath)

    if video_filepath.suffix not in video_file_extensions:
        sys.stderr.write("[!] Error: {}'s extension is not a video file type."
                         .format(video_filepath.name))
        sys.exit()


def validate_frame_order(start, end):
    """Ensure that ordering of start/end values are correct"""

    if not end > start > -1:
        sys.stderr.write("Error: Start/end values not correct."
                         " (non-zero with end > start).")
        sys.exit()


def validate_frame_file(frame_dir, frame_number):
    if not glob(str(frame_dir/"*"/("*_"+str(frame_number)+"_*"+".png"))):
        sys.stderr.write("Error: Frame {} does not point to valid file."
                         .format(frame_number))
        sys.exit()


def validate_frame_range(frame_dir, start, end):
    """Validate if start and end frame numbers point to valid frame
    files."""

    validate_directory(frame_dir)
    validate_frame_order(start, end)
    validate_frame_file(frame_dir, start)
    validate_frame_file(frame_dir, end)


###############################################################################
#                  VIDEO/FRAME READING FUNCTIONS BEGIN HERE                   #
###############################################################################


def get_video_properties(filepath):
    vidcap = cv2.VideoCapture(str(filepath))

    properties = {
        "fps": vidcap.get(cv2.CAP_PROP_FPS)
    }

    return properties


def get_first_video_frame(filepath):
    vidcap = cv2.VideoCapture(str(filepath))
    retval, frame = vidcap.read()
    vidcap.release()

    return frame


def get_frame_from_file(path, frame_number):
    """Generic function for reading a numbered frame from a file.
    Assumes that frames are stored within a subfolder. Ideally, should
    be rewritten w/ regular expressions to make subfolders optional."""

    frame_list = glob(str(path/"*"/("*_" + str(frame_number) + "_*.png")))
    frame = cv2.imread(frame_list[0])

    return frame


class FrameReader:
    """An alternative to OpenCV's VideoCapture class. Fetches frames
    from image files rather than a video file. This allows tests
    to be run on specific sequences of frames if they're extracted
    ahead of time, which is useful for hours-long video files."""

    def __init__(self, frame_dir, fps, start, end):
        self.frame_dir = frame_dir
        self.frame_path_list = glob(str(self.frame_dir/"**"/"*.png"),
                                    recursive=True)
        p = re.compile(r'.*_(\d+)_.*')
        self.frame_path_dict = {m.group(1): m.group(0) for m in
                                [p.match(s) for s in self.frame_path_list]}
        self.fps = fps

        self.start_frame = start
        self.end_frame = end
        self.total_frames = end - start + 1

        self.frames_read = 0
        self.next_frame_number = self.start_frame
        self.frame_shape = None

    def get_filepath(self, frame_number):
        return self.frame_path_dict[str(frame_number)]

    def get_frame(self):
        if self.next_frame_number <= self.end_frame:
            frame_number = self.next_frame_number
            timestamp = self.frame_number_to_timestamp(frame_number)

            filepath = self.get_filepath(frame_number)
            frame = cv2.imread(filepath)

            if frame.data:
                self.frame_shape = frame.shape
                self.frames_read += 1
                self.next_frame_number += 1

        # This is for the case when frames are requested in batches of N, but
        # total_frames is not a multiple of N. In that case, self.end_frame
        # will eventually be exceeded, so return empty values.
        else:
            frame = np.zeros(self.frame_shape).astype(np.uint8)
            frame_number = -1
            timestamp = "00:00:00.000"

        return frame, frame_number, timestamp

    def get_n_frames(self, n):
        frames, frame_numbers, timestamps = [], [], []
        for _ in range(n):
            frame, frame_number, timestamp = self.get_frame()

            frames.append(frame)
            frame_numbers.append(frame_number)
            timestamps.append(timestamp)

        return frames, frame_numbers, timestamps

    def frame_number_to_timestamp(self, frame_number):
        total_s = frame_number / self.fps
        timestamp = pd.Timestamp("00:00:00.000") + pd.Timedelta(total_s, 's')

        return timestamp


class VideoReader(cv2.VideoCapture):
    def __init__(self, video_filepath):
        super(VideoReader, self).__init__(str(video_filepath))

        self.total_frames = int(self.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames_read = 0
        self.frame_shape = None

    def get_frame(self):
        if self.get(cv2.CAP_PROP_POS_FRAMES) <= self.total_frames:
            frame_number = int(self.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = self.ms_to_ts(self.get(cv2.CAP_PROP_POS_MSEC))
            success, frame = self.read()

            if success:
                self.frame_shape = frame.shape
                self.frames_read += 1

        # This is for the case when frames are requested in batches of N, but
        # total_frames is not a multiple of N. In that case, self.end_frame
        # will eventually be exceeded, so return empty values.
        else:
            frame = np.zeros(self.frame_shape).astype(np.uint8)
            frame_number = -1
            timestamp = "00:00:00.000"

        return frame, frame_number, timestamp

    def get_n_frames(self, n):
        frames, frame_numbers, timestamps = [], [], []
        for _ in range(n):
            frame, frame_number, timestamp = self.get_frame()

            frames.append(frame)
            frame_numbers.append(frame_number)
            timestamps.append(timestamp)

        return frames, frame_numbers, timestamps

    def ms_to_ts(self, ms):
        return pd.Timestamp("00:00:00.000") + pd.Timedelta(ms, 'ms')


###############################################################################
#               RESEARCH EXPERIMENTATION FUNCTIONS BEGIN HERE                 #
###############################################################################


def get_corners_from_file(parent_directory):
    """Non-GUI alternative to get_video_attributes for time saving."""

    with open(str(parent_directory / "attributes.json")) as json_file:
        video_attributes = json.load(json_file)

        # Convert from string to individual integer values
        video_attributes["corners"] \
            = [(int(video_attributes["corners"][0][0]),
                int(video_attributes["corners"][0][1])),
               (int(video_attributes["corners"][1][0]),
                int(video_attributes["corners"][1][1]))]

    return video_attributes["corners"]


def generate_test_dir(parent_dir):
    """Generate test directory based on the following scheme:
        parent_dir/<today's date>/<ID of last test + 1>

    If no test has been run today, set ID to 1."""

    # Set base testing directory to today's date
    date_dir = parent_dir / str(date.today())

    if not date_dir.exists():
        # Date directory doesnt exist, so must be first test run today
        test_dir = date_dir / "1"

    else:
        # Fetch names of all subdirectories in date_dir, then get the max
        last_test_id = max([int(path.stem) for path in
            [Path(path_str) for path_str in glob(str(date_dir / "*/"))]])

        # Set test directory to last test incremented by one
        test_dir = date_dir / str(last_test_id + 1)

    return test_dir