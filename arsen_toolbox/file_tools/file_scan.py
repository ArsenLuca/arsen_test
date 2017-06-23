# -*- coding: utf-8 -*-

import os

def make_dir(new_dir):
    """ make a directory """
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def list_file(root, recursive, exts):
    """ list specific files in root """
    i = 0
    if recursive:
        cat = {}
        for path, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files.sort()
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and ((exts is None) or (suffix in exts)):
                    if path not in cat:
                        cat[path] = len(cat)
                    yield (i, os.path.relpath(fpath, root), fpath, cat[path])
                    i += 1
        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            print(os.path.relpath(k, root), v)
    else:
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and ((exts is None) or (suffix in exts)):
                yield (i, os.path.relpath(fpath, root), fpath, 0)
                i += 1

def test_list_file():
    """ test list file """
    file_list = list_file(root="C:/images/top100/train_append/n04081281", \
    recursive=False, exts=[".jpg"])
    print list(file_list)
