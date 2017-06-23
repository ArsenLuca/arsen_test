# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys

#curr_path = os.path.abspath(os.path.dirname(__file__))
#sys.path.append(os.path.join(curr_path, "../python"))
import mxnet as mx
import random
import argparse
import cv2
import time
import traceback
from easydict import EasyDict as edict


def list_image(root, recursive, exts):
    i = 0
    if recursive:
        cat = {}
        for path, subdirs, files in os.walk(root, followlinks=True):
            subdirs.sort()
            print(len(cat), path)
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    if path not in cat:
                        cat[path] = len(cat)
                    yield (i, os.path.relpath(fpath, root), cat[path])
                    i += 1
    else:
        for fname in os.listdir(root):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                yield (i, os.path.relpath(fpath, root), 0)
                i += 1

# for MIT67 Indoor Dataset
# set train: 1, test:0, val: -1
def list_image_const_num_per_category(root, recursive, exts,shuffle,train_num,test_num):
    i = 0
    if recursive:
        cat = {}
        for path, subdirs, files in os.walk(root, followlinks=True):
            subdirs.sort()
            print(len(cat), path)
            
            if shuffle is True:# by zys
                random.seed(100)
                random.shuffle(files)
            j = 0    
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    if path not in cat:
                        cat[path] = len(cat)
                    if j < train_num:
                        yield(1, i, os.path.relpath(fpath, root), cat[path])
                    elif j < train_num + test_num:
                        yield(0, i, os.path.relpath(fpath, root), cat[path])
                    else:
                        yield(-1, i, os.path.relpath(fpath, root), cat[path])
                    j += 1
                    i += 1

def write_list(path_out, image_list):
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            line = '%d\t' % item[0]
            for j in item[2:]:
                line += '%f\t' % j
            line += '%s\n' % item[1]
            fout.write(line)

def make_list(args):
    args = set_default(args)

    image_list = list_image(args.root, args.recursive, args.exts)
    image_list = list(image_list)
    if args.shuffle is True:
        random.seed(100)
        random.shuffle(image_list)
    N = len(image_list)
    chunk_size = (N + args.chunks - 1) / args.chunks
    for i in xrange(args.chunks):
        chunk = image_list[i * chunk_size:(i + 1) * chunk_size]
        if args.chunks > 1:
            str_chunk = '_%d' % i
        else:
            str_chunk = ''
        sep = int(chunk_size * args.train_ratio)
        sep_test = int(chunk_size * args.test_ratio)
        write_list(args.prefix + str_chunk + '_test.lst', chunk[:sep_test])
        write_list(args.prefix + str_chunk + '_train.lst', chunk[sep_test:sep_test + sep])
        write_list(args.prefix + str_chunk + '_val.lst', chunk[sep_test + sep:])

def make_list_const_num_per_category(args): # for MIT67 Indoor Dataset

    args = set_default(args)
    image_list = list_image_const_num_per_category(args.root, args.recursive, args.exts,args.shuffle,args.train_num,args.test_num)
    image_list = list(image_list)
    if args.shuffle is True:
        random.seed(100)
        random.shuffle(image_list)

    N = len(image_list)
    chunk_size = (N + args.chunks - 1) / args.chunks
    
    list_train = []
    list_test = []
    list_val = []
    
    for i in xrange(args.chunks):
        chunk = image_list[i * chunk_size:(i + 1) * chunk_size]
        if args.chunks > 1:
            str_chunk = '_%d' % i
        else:
            str_chunk = ''
        for id_image in xrange(len(chunk)):
            if chunk[id_image][0] is 1:
                list_train.append(chunk[id_image][1:])
            elif chunk[id_image][0] is 0:
                list_test.append(chunk[id_image][1:])
            else:
                list_val.append(chunk[id_image][1:])
                
        write_list(args.prefix + str_chunk + '_test.lst', list_test)
        write_list(args.prefix + str_chunk + '_train.lst', list_train)
        write_list(args.prefix + str_chunk + '_val.lst', list_val)
        
def read_list(path_in):
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            yield item

def image_encode(args, item, q_out):
    fullpath = os.path.join(args.root, item[1])
    try:
        img = cv2.imread(fullpath, args.color)
    except:
        traceback.print_exc()
        print('imread error trying to load file: %s ' % fullpath)
        return
    if img is None:
        print('imread read blank (None) image for file: %s' % fullpath)
        return
    if args.center_crop:
        if img.shape[0] > img.shape[1]:
            margin = (img.shape[0] - img.shape[1]) / 2;
            img = img[margin:margin + img.shape[1], :]
        else:
            margin = (img.shape[1] - img.shape[0]) / 2;
            img = img[:, margin:margin + img.shape[0]]
    if args.resize:
        if img.shape[0] > img.shape[1]:
            newsize = (args.resize, img.shape[0] * args.resize / img.shape[1])
        else:
            newsize = (img.shape[1] * args.resize / img.shape[0], args.resize)
        img = cv2.resize(img, newsize)
    if len(item) > 3 and args.pack_label:
        header = mx.recordio.IRHeader(0, item[2:], item[0], 0)
    else:
        header = mx.recordio.IRHeader(0, item[2], item[0], 0)

    try:
        s = mx.recordio.pack_img(header, img, quality=args.quality, img_fmt=args.encoding)
        q_out.put((s, item))
    except Exception:
        traceback.print_exc()
        print('pack_img error on file: %s' % fullpath)
        return

def read_worker(args, q_in, q_out):
    while True:
        item = q_in.get()
        if item is None:
            break
        image_encode(args, item, q_out)

def write_worker(q_out, fname, working_dir):
    pre_time = time.time()
    count = 0
    fname_rec = os.path.basename(fname)
    fname_rec = os.path.splitext(fname)[0] + '.rec'
    fout = open(fname+'.tmp', 'w')
    record = mx.recordio.MXRecordIO(os.path.join(working_dir, fname_rec), 'w')
    while True:
        deq = q_out.get()
        if deq is None:
            break
        s, item = deq
        record.write(s)

        line = '%d\t' % item[0]
        for j in item[2:]:
            line += '%f\t' % j
        line += '%s\n' % item[1]
        fout.write(line)

        if count % 1000 == 0:
            cur_time = time.time()
            print('time:', cur_time - pre_time, ' count:', count)
            pre_time = cur_time
        count += 1
    fout.close()
    os.remove(fname)
    os.rename(fname+'.tmp', fname)

def set_default(args):

    args.prefix = os.path.abspath(args.prefix)
    args.root = os.path.abspath(args.root)

    if not args.has_key('list'): args.list = False
    if not args.has_key('exts'): args.exts = ['.jpeg', '.jpg']
    if not args.has_key('chunks'): args.chunks = 1
    if not args.has_key('recursive'): args.recursive = False

    if not args.has_key('resize'): args.resize = 0
    if not args.has_key('center_crop'): args.center_crop = False
    if not args.has_key('quality'): args.quality = 80
    if not args.has_key('num_thread'): args.num_thread = 1
    if not args.has_key('color'): args.color = 1
    if not args.has_key('encoding'): args.encoding = '.jpg'
    if not args.has_key('shuffle'): args.shuffle = True
    if not args.has_key('pack_label'): args.pack_label = False

    return args

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('prefix', help='prefix of input/output lst and rec files.')
    parser.add_argument('root', help='path to folder containing images.')

    cgroup = parser.add_argument_group('Options for creating image lists')
    cgroup.add_argument('--list', type=bool, default=False,
                        help='If this is set im2rec will create image list(s) by traversing root folder\
        and output to <prefix>.lst.\
        Otherwise im2rec will read <prefix>.lst and create a database at <prefix>.rec')
    cgroup.add_argument('--exts', type=list, default=['.jpeg', '.jpg'],
                        help='list of acceptable image extensions.')
    cgroup.add_argument('--chunks', type=int, default=1, help='number of chunks.')
    cgroup.add_argument('--train-ratio', type=float, default=1.0,
                        help='Ratio of images to use for training.')
    cgroup.add_argument('--test-ratio', type=float, default=0,
                        help='Ratio of images to use for testing.')
    cgroup.add_argument('--recursive', type=bool, default=False,
                        help='If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.')

    rgroup = parser.add_argument_group('Options for creating database')
    rgroup.add_argument('--resize', type=int, default=0,
                        help='resize the shorter edge of image to the newsize, original images will\
        be packed by default.')
    rgroup.add_argument('--center-crop', type=bool, default=False,
                        help='specify whether to crop the center image to make it rectangular.')
    rgroup.add_argument('--quality', type=int, default=80,
                        help='JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9')
    rgroup.add_argument('--num-thread', type=int, default=1,
                        help='number of thread to use for encoding. order of images will be different\
        from the input list if >1. the input list will be modified to match the\
        resulting order.')
    rgroup.add_argument('--color', type=int, default=1, choices=[-1, 0, 1],
                        help='specify the color mode of the loaded image.\
        1: Loads a color image. Any transparency of image will be neglected. It is the default flag.\
        0: Loads image in grayscale mode.\
        -1:Loads image as such including alpha channel.')
    rgroup.add_argument('--encoding', type=str, default='.jpg', choices=['.jpg', '.png'],
                        help='specify the encoding of the images.')
    rgroup.add_argument('--shuffle', default=True, help='If this is set as True, \
        im2rec will randomize the image order in <prefix>.lst')
    rgroup.add_argument('--pack-label', default=False,
        help='Whether to also pack multi dimensional label in the record file') 
    args = parser.parse_args()
    args.prefix = os.path.abspath(args.prefix)
    args.root = os.path.abspath(args.root)
    return args


def make_rec(args):
    args = set_default(args)

    if os.path.isdir(args.prefix):
        working_dir = args.prefix
    else:
        working_dir = os.path.dirname(args.prefix)
    files = [os.path.join(working_dir, fname) for fname in os.listdir(working_dir)
                if os.path.isfile(os.path.join(working_dir, fname))]
    count = 0
    for fname in files:
        if fname.startswith(args.prefix) and fname.endswith('.lst'):
            print('Creating .rec file from', fname, 'in', working_dir)
            count += 1
            image_list = read_list(fname)
            # -- write_record -- #
            try:
                import multiprocessing
                q_in = [multiprocessing.Queue(1024) for i in range(args.num_thread)]
                q_out = multiprocessing.Queue(1024)
                read_process = [multiprocessing.Process(target=read_worker, args=(args, q_in[i], q_out)) \
                                for i in range(args.num_thread)]
                for p in read_process:
                    p.start()
                write_process = multiprocessing.Process(target=write_worker, args=(q_out, fname, working_dir))
                write_process.start()

                for i, item in enumerate(image_list):
                    q_in[i % len(q_in)].put(item)
                for q in q_in:
                    q.put(None)
                for p in read_process:
                    p.join()

                q_out.put(None)
                write_process.join()
            except ImportError:
                print('multiprocessing not available, fall back to single threaded encoding')
                import Queue
                q_out = Queue.Queue()
                fname_rec = os.path.basename(fname)
                fname_rec = os.path.splitext(fname)[0] + '.rec'
                record = mx.recordio.MXRecordIO(os.path.join(working_dir, fname_rec), 'w')
                cnt = 0
                pre_time = time.time()
                for item in image_list:
                    image_encode(args, item, q_out)
                    if q_out.empty():
                        continue
                    _, s, _ = q_out.get()
                    record.write(s)
                    if cnt % 1000 == 0:
                        cur_time = time.time()
                        print('time:', cur_time - pre_time, ' count:', cnt)
                        pre_time = cur_time
                    cnt += 1
    if not count:
        print('Did not find and list file with prefix %s'%args.prefix)

def gen_dataset_rec(lst_path, image_path, **params):
    """ generate .lst & .rec files for dataset """
    # make .lst
    if params["fix_ratio"]:

        edicts_lst = edict({'prefix':lst_path, 'root':image_path, 'train_ratio':params["train_ratio"], \
                            'test_ratio':params["test_ratio"], 'recursive':True})
        make_list(edicts_lst)
    else:
        edicts_lst = edict({'prefix':lst_path, 'root':image_path, 'train_num':params["train_num"], \
                            'test_num':params["test_num"], 'recursive':True})
        make_list_const_num_per_category(edicts_lst)

    # make .rec
    edicts_rec = edict({'prefix':lst_path, 'root':image_path, 'resize':256})
    make_rec(edicts_rec)
