import os
import shutil
import math
import random

from .file_scan import list_file

def select_max_num_samples_per_category(file_list, max_num_per_cat, randomly):
    """ select samples from each category, with max_num_per_category"""
    count_dict = dict()
    if randomly:
        random.shuffle(file_list) 

    for file_info in file_list:
        if not count_dict.get(file_info[-1]):
            count_dict[file_info[-1]] = 1
        else:
            if count_dict[file_info[-1]] >= max_num_per_cat:
                continue
            count_dict[file_info[-1]] += 1
        yield file_info[2]

def file_select(folder, exts=None, **params):
    """ select files """
    file_list = list_file(root=folder, recursive=True, exts=exts)

    if params.get("max_num_per_cat"):
        file_list_selected = select_max_num_samples_per_category(list(file_list), \
                                params["max_num_per_cat"], randomly=params.get("random"))
    else:
        file_list_selected = file_list

    return file_list_selected
