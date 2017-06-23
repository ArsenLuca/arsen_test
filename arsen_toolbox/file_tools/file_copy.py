# -*- coding: utf-8 -*-

import os
import shutil
import math
import collections

from .file_scan import list_file
from .file_scan import make_dir

def copying_files_in_specific_folder(folder, res_folder, exts=None):
    """ Copy all files in folder recursively into res_folder"""

    file_list = list_file(root=folder, recursive=True, exts=exts)
    make_dir(res_folder)

    for org_file in file_list:
        shutil.copy(org_file[1], res_folder)

def copying_files_in_list(file_list, res_folder):
    """ Copy all files in a list recursively into res_folder"""

    make_dir(res_folder)

    for org_file in file_list:
        shutil.copy(org_file[1], res_folder)

def copying_files(source, res_folder, exts=None):
    """ Copy all files intelligently into res_folder
    source: a folder or an Iterable object(list, tuple, dictionary, generator...)
    """

    if isinstance(source, str):
        file_list = list_file(root=folder, recursive=True, exts=exts)
        file_list = [file_info[2] for file_info in file_list]

    elif isinstance(source, collections.Iterable):
        if isinstance(source, dict):
            file_list = source.keys()
        else:
            file_list = source
    else:
        raise Exception("Invalid source")

    make_dir(res_folder)
    for org_file in file_list:
        shutil.copy(org_file, res_folder) 
