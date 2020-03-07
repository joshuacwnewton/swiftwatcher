import swiftwatcher.ui as ui

filepaths = ui.select_filepaths()
for filepath in filepaths:
    ui.save_corners_to_file(filepath)
