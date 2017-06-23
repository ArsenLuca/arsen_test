# -*- coding: utf-8 -*-

import os
import shutil
import math

from .file_scan import list_file

def auto_rename(folder_path, exts, name_format, new_ext=None):
    """ auto-rename files in a specific folder 
        name_format: "Google_download_cat_{id}"
        new_ext: if None, keep original extention
    """
    file_list = list(list_file(root=folder, recursive=False, exts=exts))
    format_precision = int(math.log10(len(file_list))) + 1
    precision_str = "%0{}d".format(format_precision)

    for i, image_path in enumerate(file_list):
        org_filename, org_ext = os.path.splitext(image_path)
        new_ext = org_ext if new_ext is None

        new_image_path = os.path.join(org_filename, name_format.format(id=precision_str % i) + new_ext)

        while os.path.exits(new_image_path):
            precision_str += "_1"
            new_image_path = os.path.join(org_filename, name_format.format(id=precision_str % i) + new_ext)

        os.rename(image_path, new_image_path)