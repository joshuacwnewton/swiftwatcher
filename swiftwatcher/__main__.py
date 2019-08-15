# Algorithm components
import swiftwatcher.video_processing as vid
import swiftwatcher.data_analysis as data

# File I/O
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog


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
            '.bdm',
            '.bdmv', '.bdt2', '.bdt3', '.bik', '.bin', '.bix',
            '.bmk', '.bnp', '.box', '.bs4', '.bsf', '.bvr', '.byu', '.camproj',
            '.camrec', '.camv', '.ced', '.cel', '.cine', '.cip',
            '.clpi', '.cmmp', '.cmmtpl', '.cmproj', '.cmrec', '.cpi', '.cst',
            '.cvc', '.cx3', '.d2v', '.d3v', '.dat', '.dav', '.dce',
            '.dck', '.dcr', '.dcr', '.ddat', '.dif', '.dir', '.divx', '.dlx',
            '.dmb', '.dmsd', '.dmsd3d', '.dmsm', '.dmsm3d', '.dmss',
            '.dmx', '.dnc', '.dpa', '.dpg', '.dream', '.dsy', '.dv', '.dv-avi',
            '.dv4', '.dvdmedia', '.dvr', '.dvr-ms', '.dvx', '.dxr',
            '.dzm', '.dzp', '.dzt', '.edl', '.evo', '.eye', '.ezt', '.f4p',
            '.f4v',
            '.fbr', '.fbr', '.fbz', '.fcp', '.fcproject',
            '.ffd', '.flc', '.flh', '.fli', '.flv', '.flx', '.gfp', '.gl',
            '.gom',
            '.grasp', '.gts', '.gvi', '.gvp', '.h264', '.hdmov',
            '.hkm', '.ifo', '.imovieproj', '.imovieproject', '.ircp', '.irf',
            '.ism', '.ismc', '.ismv', '.iva', '.ivf', '.ivr', '.ivs',
            '.izz', '.izzy', '.jss', '.jts', '.jtv', '.k3g', '.kmv', '.ktn',
            '.lrec', '.lsf', '.lsx', '.m15', '.m1pg', '.m1v', '.m21',
            '.m21', '.m2a', '.m2p', '.m2t', '.m2ts', '.m2v', '.m4e', '.m4u',
            '.m4v', '.m75', '.mani', '.meta', '.mgv', '.mj2', '.mjp',
            '.mjpg', '.mk3d', '.mkv', '.mmv', '.mnv', '.mob', '.mod', '.modd',
            '.moff', '.moi', '.moov', '.mov', '.movie', '.mp21',
            '.mp21', '.mp2v', '.mp4', '.mp4v', '.mpe', '.mpeg', '.mpeg1',
            '.mpeg4',
            '.mpf', '.mpg', '.mpg2', '.mpgindex', '.mpl',
            '.mpl', '.mpls', '.mpsub', '.mpv', '.mpv2', '.mqv', '.msdvd',
            '.mse',
            '.msh', '.mswmm', '.mts', '.mtv', '.mvb', '.mvc',
            '.mvd', '.mve', '.mvex', '.mvp', '.mvp', '.mvy', '.mxf', '.mxv',
            '.mys', '.ncor', '.nsv', '.nut', '.nuv', '.nvc', '.ogm',
            '.ogv', '.ogx', '.osp', '.otrkey', '.pac', '.par', '.pds', '.pgi',
            '.photoshow', '.piv', '.pjs', '.playlist', '.plproj',
            '.pmf', '.pmv', '.pns', '.ppj', '.prel', '.pro', '.prproj',
            '.prtl',
            '.psb', '.psh', '.pssd', '.pva', '.pvr', '.pxv',
            '.qt', '.qtch', '.qtindex', '.qtl', '.qtm', '.qtz', '.r3d', '.rcd',
            '.rcproject', '.rdb', '.rec', '.rm', '.rmd', '.rmd',
            '.rmp', '.rms', '.rmv', '.rmvb', '.roq', '.rp', '.rsx', '.rts',
            '.rts',
            '.rum', '.rv', '.rvid', '.rvl', '.sbk', '.sbt',
            '.scc', '.scm', '.scm', '.scn', '.screenflow', '.sec', '.sedprj',
            '.seq', '.sfd', '.sfvidcap', '.siv', '.smi', '.smi',
            '.smil', '.smk', '.sml', '.smv', '.spl', '.sqz', '.srt', '.ssf',
            '.ssm', '.stl', '.str', '.stx', '.svi', '.swf', '.swi',
            '.swt', '.tda3mt', '.tdx', '.thp', '.tivo', '.tix', '.tod', '.tp',
            '.tp0', '.tpd', '.tpr', '.trp', '.ts', '.tsp', '.ttxt',
            '.tvs', '.usf', '.usm', '.vc1', '.vcpf', '.vcr', '.vcv', '.vdo',
            '.vdr', '.vdx', '.veg', '.vem', '.vep', '.vf', '.vft',
            '.vfw', '.vfz', '.vgz', '.vid', '.video', '.viewlet', '.viv',
            '.vivo',
            '.vlab', '.vob', '.vp3', '.vp6', '.vp7', '.vpj',
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
                                                    title='Choose a file')
                filepaths = (filepaths +
                             ([Path(f) for f in list(root.tk.splitlist(files))
                              if is_video_file(Path(f).suffix)]))
                filenames = [f.stem for f in filepaths]
                print("Video files to be analysed: ")
                print(*filenames, sep="\n")
                ipt = input("\nAre there additional files you would like to "
                            "select? (Y/N) \n"
                            "Input: ")
                if (ipt is not "y") and (ipt is not "Y"):
                    break
        except TypeError:
            print("[!] No video directory selected.")

        return filepaths

    def create_config_list(filepaths):
        """Load a config file (or create one if it doesn't exist)
        corresponding to each video filepath."""

        config_list = []
        if len(filepaths) is 0:
            print("[!] No videos to analyse. Please try again.")
        else:
            ipt = input("\nWould you like to re-use the first video's "
                        "corners for each video? (Y/N) \n"
                        "Input: ")

            for filepath in filepaths:
                # "Result" csv files will also be stored in this directory
                base_dir = filepath.parent/filepath.stem
                if not base_dir.exists():
                    base_dir.mkdir(parents=True, exist_ok=True)

                config_filepath = base_dir/(filepath.stem + ".json")

                # Create config file
                config = {
                    "name": filepath.name,
                    "timestamp": "00:00:00.000000",
                    "src_filepath": filepath,
                    "base_dir": base_dir
                }
                if (len(config_list) > 0) and (ipt == "y" or ipt == "Y"):
                    config["corners"] = config_list[0]["corners"]
                else:
                    config["corners"] = vid.select_corners(filepath)

                config["src_filepath"] = filepath
                config["base_dir"] = base_dir

                config_list.append(config)

        return config_list

    video_paths = fetch_video_filepaths()
    configs = create_config_list(video_paths)

    return configs


def main():
    """Execute each of the core functions of the swift-counting algorithm."""

    configs = load_configs()

    for config in configs:
        # "Events" are possible entering occurrences which must be classified
        events = vid.swift_counting_algorithm(config)

        if len(events) > 0:
            features = data.generate_feature_vectors(events)
            labels = data.generate_classifications(features)
            data.export_results(config, labels)

        else:
            print("No detected chimney swifts in specified video.")


if __name__ == "__main__":
    main()
