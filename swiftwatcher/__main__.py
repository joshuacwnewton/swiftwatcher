# Algorithm components
import swiftwatcher.video_processing as vid
import swiftwatcher.data_analysis as data
import utils.gui as gui

# File I/O
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# Import for swift_counting_algorithm
import pandas as pd
import cv2
import sys


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


def swift_counting_algorithm(config):
    """Full algorithm which uses FrameQueue methods to process an entire video
    from start to finish."""

    def create_dataframe(passed_list):
        """Convert list of events to pandas dataframe."""
        dataframe = pd.DataFrame(passed_list,
                                 columns=list(passed_list[0].keys())
                                 ).astype('object')
        dataframe["TMSTAMP"] = pd.to_datetime(dataframe["TMSTAMP"])
        dataframe["TMSTAMP"] = dataframe["TMSTAMP"].dt.round('us')
        dataframe.set_index(["TMSTAMP", "FRM_NUM"], inplace=True)

        return dataframe

    print("[*] Now processing {}.".format(config["name"]))
    # print("[-]     Status updates will be given every 100 frames.")

    fq = vid.FrameQueue(config)
    while fq.frames_processed < fq.src_framecount:
        success = False

        # Store state variables in case video processing glitch occurs
        # (e.g. due to poorly encoded video)
        pos = fq.stream.get(cv2.CAP_PROP_POS_FRAMES)
        read = fq.frames_read
        proc = fq.frames_processed

        try:
            # Load frames until queue is filled
            if fq.frames_read < (fq.queue_size - 1):
                success = fq.load_frame()
                fq.preprocess_frame()
                # fq.segment_frame() (not needed until queue is filled)
                # fq.match_segments() (not needed until queue is filled)
                # fq.analyse_matches() (not needed until queue is filled)

            # Process queue full of frames
            elif (fq.queue_size - 1) <= fq.frames_read < fq.src_framecount:
                success = fq.load_frame()
                fq.preprocess_frame()
                fq.segment_frame()
                fq.match_segments()
                fq.analyse_matches()

            # Load blank frames until queue is empty
            elif fq.frames_read == fq.src_framecount:
                success = fq.load_frame(blank=True)
                # fq.preprocess_frame() (not needed for blank frame)
                fq.segment_frame()
                fq.match_segments()
                fq.analyse_matches()

        except Exception as e:
            # TODO: Print statements here should be replaced with logging
            # Previous: print("[!] Error has occurred, see: '{}'.".format(e))
            # I'm not satisfied with how unexpected errors are handled
            fq.failcount += 1

            # Increment state variables to ensure algorithm doesn't get stuck
            if fq.stream.get(cv2.CAP_PROP_POS_FRAMES) == pos:
                fq.stream.grab()
            if fq.frames_read == read:
                fq.frames_read += 1
            if fq.frames_processed == proc:
                fq.frames_processed += 1

        if success:
            fq.failcount = 0
        else:
            fq.failcount += 1

        # Break if too many sequential errors
        if fq.failcount >= 10:
            # TODO: Print statements here should be replaced with logging
            # Previous: print("[!] Too many sequential errors have occurred. "
            #                 "Halting algorithm...")
            # I'm not satisfied with how unexpected errors are handled
            fq.frames_processed = fq.src_framecount + 1

        # Status updates
        if fq.frames_processed % 25 is 0 and fq.frames_processed is not 0:
            sys.stdout.write("\r[-]     {0}/{1} frames processed.".format(
                fq.frames_processed, fq.src_framecount))
            sys.stdout.flush()

    if fq.event_list:
        df_eventinfo = create_dataframe(fq.event_list)
    else:
        df_eventinfo = []
    print("")

    return df_eventinfo


def main():
    """Execute each of the core functions of the swift-counting algorithm."""

    configs = load_configs()

    for config in configs:
        if len(config["corners"]) == 2:
            events = swift_counting_algorithm(config)

            if len(events) > 0:
                features = data.generate_feature_vectors(events)
                labels = data.generate_classifications(features)
                total = data.export_results(config, labels)
                print("[-]     Analysis complete. {} detected chimney swifts "
                      "in specified video.".format(total))
            else:
                print("[-]     Analysis complete. No detected chimney swifts "
                      "in specified video.")
        else:
            print("[!] Corners not selected for {}. Cannot process."
                  .format(config["name"]))


if __name__ == "__main__":
    main()
