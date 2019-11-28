from utils.video_io import gui_select_files, create_config_file

filepaths = gui_select_files()
for filepath in filepaths:
    create_config_file(filepath)
