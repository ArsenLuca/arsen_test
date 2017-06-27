# -*- coding: utf-8 -*-
import os
import datetime

def get_file_name_by_time(prefix="./", ext=""):
    """ get a file name by time now """
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    return os.path.join(prefix, datetime.datetime.now().strftime("%y%m%d%h%m%s") + ext)
