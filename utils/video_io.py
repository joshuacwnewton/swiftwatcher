from swiftwatcher import gui as gui

import cv2

from pathlib import Path
from os import fspath
import datetime

import tkinter as tk
from tkinter import filedialog

import argparse
import sys

import json


def gui_select_files():
    """Select files using tkinter gui and return their paths."""

    filepaths = []
    try:
        root = tk.Tk()
        root.withdraw()
        # /\ See: https://stackoverflow.com/questions/1406145/
        while True:
            # GUI file selection
            files = filedialog.askopenfilenames(parent=root,
                                                title='Choose the files '
                                                      'you wish to '
                                                      'analyse.')

            # Convert selected files (unique) into Path objects
            filepaths = (filepaths +
                         ([Path(f) for f in list(root.tk.splitlist(files))
                           if Path(f) not in filepaths]))

            # Query the user if they're happy with the files
            filenames = ["[-]     {}".format(f.name) for f in filepaths]
            print("[*] Video files to be analysed: ")
            print(*filenames, sep="\n")
            ipt = input("[*] Are there additional files you would like to "
                        "select? (Y/N) \n"
                        "[-]     Input: ")
            if (ipt is not "y") and (ipt is not "Y"):
                break
    except TypeError:
        print("[!] No video directory selected.")

    return filepaths


def parse_filepaths():
    """Parse all command line arguments as filepaths."""

    parser = argparse.ArgumentParser()
    parser.add_argument("filepaths", nargs="*")
    args = parser.parse_args()

    args.filepaths = [Path(filepath) for filepath in args.filepaths]

    return args.filepaths


def validate_filepath(filepath):
    """Ensure that file path points to a valid file."""

    # Ensure that path is Path object
    filepath = Path(filepath)

    if not Path.is_file(filepath):
        sys.stderr.write("Error: {} does not point to a valid file."
                         .format(filepath.name))
        sys.exit()


def validate_video_extension(video_filepath):
    """Ensure that extension of file is a video file type."""

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
        sys.stderr.write("Error: {}'s extension is not a video file type."
                         .format(video_filepath.name))
        sys.exit()


def extract_video_frames(video_filepath,
                         base_dir,
                         subdir_name_scheme,
                         file_name_scheme):
    """Extract frames from video file and save them to image files
    according to filename and dirname schemes.

    Schemes are passed as list of either string literals or variables
    from a limited list of values:
        -video_filename
        -frame_number
        -datetime class attributes
            -year
            -month
            -day
            -hour
            -minute
            -second
            -millisecond
            -microsecond"""

    # Create initial datetime object
    start_time = datetime.datetime(100, 1, 1, 0, 0, 0, 0)

    # Ensure that video filepath is a Path object
    base_dir = Path(base_dir)
    video_filepath = Path(video_filepath)
    video_filename = video_filepath.name

    stream = cv2.VideoCapture(str(video_filepath))
    while True:
        # Determine attributes associated with frame
        frame_number = int(stream.get(cv2.CAP_PROP_POS_FRAMES))
        frame_ms = int(stream.get(cv2.CAP_PROP_POS_MSEC))
        timestamp = start_time + datetime.timedelta(milliseconds=frame_ms)

        # Attempt to load new frame
        success, frame = stream.read()
        if not success:
            break

        # Create sub directory string
        sub_dir = ""
        for item in subdir_name_scheme:
            if item[0] == "%" and item[-1] == "%":
                if item in ["%hour%", "%minute%", "%second%"]:
                    num = eval("timestamp." + item.replace("%", ""))
                    sub_dir += ("{:02d}".format(num))
                elif item in ["%microsecond%"]:
                    num = eval("timestamp." + item.replace("%", ""))
                    sub_dir += ("{:03d}".format(num))
                elif item in ["%video_filename%", "%frame_number%"]:
                    sub_dir += str((eval(item.replace("%", ""))))
                else:
                    sys.stderr.write("Error: {} not valid for naming scheme."
                                     .format(item))
            else:
                sub_dir += item

        # Create file name string
        file_name = ""
        for item in file_name_scheme:
            if item[0] == "%" and item[-1] == "%":
                if item in ["%hour%", "%minute%", "%second%"]:
                    num = eval("timestamp." + item.replace("%", ""))
                    file_name += "{:02d}".format(num)
                elif item in ["%microsecond%"]:
                    num = eval("timestamp." + item.replace("%", ""))
                    file_name += "{:06d}".format(num)
                elif item in ["%video_filename%", "%frame_number%"]:
                    file_name += str((eval(item.replace("%", ""))))
                else:
                    sys.stderr.write("Error: {} not valid for naming scheme."
                                     .format(item))
            else:
                file_name += item

        # Create output_directory if it doesn't exist
        output_dir = base_dir / sub_dir
        if not output_dir.exists():
            Path.mkdir(output_dir, parents=True)

        # Write frame to output_directory
        cv2.imwrite(str(output_dir / (file_name + ".png")), frame)


def parse_filepath_and_framerange():
    """Parse named arguments for filepath, starting frame, and ending
    frame.."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()

    args.filepath = Path(args.filepath)

    return args.filepath, int(args.start), int(args.end)


def validate_framerange(frame_dir, start, end):
    """Validate if start and end frame numbers point to valid frame
    files."""


def create_config_file(filepath):
    base_dir = filepath.parent / filepath.stem / "frames"
    if not base_dir.exists():
        Path.mkdir(base_dir, parents=True)

    l_x = input("[*] Input the x coordinate of the left corner: ")
    l_y = input("[*] Input the y coordinate of the left corner: ")
    r_x = input("[*] Input the x coordinate of the right corner: ")
    r_y = input("[*] Input the y coordinate of the right corner: ")

    config_dict = {
        "name": fspath(filepath.name),
        "timestamp": "00:00:00.000000",
        "src_filepath": fspath(filepath),
        "base_dir": fspath(base_dir),
        "corners": [(l_x, l_y), (r_x, r_y)]
    }

    with open(fspath(base_dir / "config.json"), 'w') as fp:
        json.dump(config_dict, fp)


def config_from_file(base_dir):
    """Loads "config" dictionary from json file for experiments that
    load frames from files."""
    with open(fspath(base_dir / "config.json")) as json_file:
        config = json.load(json_file)
        config["base_dir"] = Path(config["base_dir"])
        config["src_filepath"] = Path(config["src_filepath"])
        config["corners"] = [(int(config["corners"][0][0]),
                              int(config["corners"][0][1])),
                             (int(config["corners"][1][0]),
                              int(config["corners"][1][1]))]

    return config


def load_configs():
    """Load config files for videos in video directory, or create+save config
    files if they do not yet exist."""

    def is_video_file(extension):
        """Check if file extension belongs to list of video file extensions."""

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

        return extension in video_file_extensions

    def fetch_video_filepaths():
        """Load list of filepaths (video files only) from user-chosen
        directory."""

        filepaths = []
        try:
            root = tk.Tk()
            root.withdraw()
            # /\ See: https://stackoverflow.com/questions/1406145/
            while True:
                files = filedialog.askopenfilenames(parent=root,
                                                    title='Choose the files '
                                                          'you wish to '
                                                          'analyse.')
                filepaths = (filepaths +
                             ([Path(f) for f in list(root.tk.splitlist(files))
                              if (is_video_file(Path(f).suffix)
                                  and Path(f) not in filepaths)]))
                filenames = ["[-]     {}".format(f.name) for f in filepaths]
                print("[*] Video files to be analysed: ")
                print(*filenames, sep="\n")
                ipt = input("[*] Are there additional files you would like to "
                            "select? (Y/N) \n"
                            "[-]     Input: ")
                if (ipt is not "y") and (ipt is not "Y"):
                    break
        except TypeError:
            print("[!] No video directory selected.")

        return filepaths

    def create_config_list(filepaths):
        """Load a config file (or create one if it doesn't exist)
        corresponding to each video filepath."""

        config_list = []
        ipt = None  # Input asking whether corners should be reused
        if len(filepaths) is 0:
            print("[!] No videos to analyse. Please try again.")
        else:
            for filepath in filepaths:
                # "Result" csv files will also be stored in this directory
                base_dir = filepath.parent/filepath.stem
                if not base_dir.exists():
                    base_dir.mkdir(parents=True, exist_ok=True)

                # Create config file
                config = {
                    "name": filepath.name,
                    "timestamp": "00:00:00.000000",
                    "src_filepath": filepath,
                    "base_dir": base_dir
                }
                if ipt == "y" or ipt == "Y":
                    config["corners"] = config_list[0]["corners"]
                else:
                    config["corners"] = gui.select_corners(filepath)
                config["src_filepath"] = filepath
                config["base_dir"] = base_dir

                config_list.append(config)

                if ((len(filepaths) > 1)
                    and (filepath == filepaths[0])
                    and (len(config["corners"]) == 2)):
                    ipt = input("[*] Would you like to re-use the first "
                                "video's corners for each video? (Y/N) \n"
                                "[-]     Input: ")

        return config_list

    video_paths = fetch_video_filepaths()
    configs = create_config_list(video_paths)

    return configs