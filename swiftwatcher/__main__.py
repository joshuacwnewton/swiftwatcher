# Algorithm components
import swiftwatcher.video_processing as vid
import swiftwatcher.data_analysis as data

# File I/O
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog


def load_config():
    """Load config files for videos in video directory, or create/save config
    files if they do not yet exist."""

    def is_video_file(extension):
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

    root = tk.Tk()
    root.withdraw()
    video_dir = Path(filedialog.askdirectory(parent=root, initialdir="/",
                                             title='Please select a directory '
                                                   'containing the videos you '
                                                   'wish to analyse.'))

    filepaths = [f for f in video_dir.iterdir()
                 if f.is_file() and is_video_file(f.suffix)]

    config_list = []
    if len(filepaths) is 0:
        print("[!] Specified directory contains no videos. Please try again.")
    else:
        for filepath in filepaths:
            config_dir = filepath.parent/filepath.stem
            if not config_dir.exists():
                config_dir.mkdir(parents=True, exist_ok=True)

            config_filepath = config_dir/(filepath.stem + ".json")
            if not config_filepath.exists():
                config = {
                    "name": filepath.name,
                    "timestamp": "00:00:00.000000",
                    "corners": vid.select_corners(filepath),
                }
                with config_filepath.open(mode="w") as write_file:
                    json.dump(config, write_file, indent=4)
            else:
                with config_filepath.open(mode="r") as read_file:
                    config = json.load(read_file)

            config["src_filepath"] = filepath
            config["base_dir"] = filepath.parent / filepath.stem
            config_list.append(config)

    return config_list


def main():
    """To understand the current configuration of the algorithm, please look
    to the following functions, which are outside of main() below:

    - args: command-line arguments, used for file I/O, set by
        if __name__ == "__main__": block of code.
    - params: algorithm parameters, used to tweak processing stages, set by
        set_parameters() function."""

    configs = load_config()
    for config in configs:

        events = vid.swift_counting_algorithm(config)
        if len(events) > 0:
            features = data.generate_feature_vectors(events)
            labels = data.generate_classifications(features)
            data.export_results(config, labels)

        else:
            print("No detected chimney swifts in specified video.")


if __name__ == "__main__":
    main()
