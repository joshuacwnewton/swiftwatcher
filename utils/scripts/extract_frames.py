import utils.video_io as vio
import sys

if len(sys.argv) > 1:
    filepaths = vio.parse_filepaths()
else:
    filepaths = vio.gui_select_files()

for filepath in filepaths:
    vio.validate_filepath(filepath)
    vio.validate_video_extension(filepath)
    vio.extract_video_frames(filepath, filepath.parent/filepath.stem/"frames",
                             ["%hour%", "h", "%minute%", "m"],
                             ["%video_filename%", "_",
                              "%frame_number%", "_",
                              "%hour%", "h", "%minute%", "m",
                              "%second%", "s", "%microsecond%", "us"])
