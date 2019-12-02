"""
    Contains functionality for video I/O, as well as video frame I/O.
"""

import sys
from pathlib import Path
from glob import glob

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
#                     VIDEO READING FUNCTIONS BEGIN HERE                      #
###############################################################################


def get_frame_from_file(path, frame_number):
    frame_list = glob(str(path/"*"/("*_" + str(frame_number) + "_*.png")))
    frame = cv2.imread(frame_list[0])

    return frame


def get_first_frame(filepath):
    vidcap = cv2.VideoCapture(str(filepath))
    retval, frame = vidcap.read()
    vidcap.release()

    return frame
