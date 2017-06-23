# from arsen_toolbox.cv.segment import test_seg
# from arsen_toolbox.directory.file_scan import test_list_file

# test_seg()
# test_list_file()

# from arsen_toolbox.tool.file_copy import copying_files_in_specific_folder

# copying_files_in_specific_folder(folder=r"d:/Dataset/Stanford_Dogs_Dataset/Images", res_folder=r"d:/Dataset/Stanford_Dogs_Dataset/Flattened_Images")
import shutil
from arsen_toolbox.tool.file_sel import file_select
from arsen_toolbox.tool.file_copy import copying_files
selected_files = file_select(folder=r"d:/Dataset/Stanford_Dogs_Dataset/Images", exts=[".jpg", ".jpeg", ".png"], max_num_per_cat = 12, random=True)

copying_files(selected_files, r"d:/Dataset/Stanford_Dogs_Dataset/Flattened_Images")