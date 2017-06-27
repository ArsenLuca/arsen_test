# -*- coding: utf-8 -*-
"""
    get drn symbols
"""
import os
import mxnet as mx
from easydict import EasyDict as edict

from arsen_toolbox.file_tools.file_auto_name import get_file_name_by_time
from arsen_toolbox.mx_tools.symbols.dilated_resnet import get_symbol
from arsen_toolbox.mx_tools.mx_train import mx_train
from arsen_toolbox.mx_tools.mx_dataiter import get_dataiter

if __name__ == "__main__":
    drn_sym = get_symbol(1000, 18, "3, 224, 224")
    # mx.viz.plot_network(drn).view()

    train_params = edict({
                    "base_lr": 0.1, "lr_factor": 0.1, "lr_factor_epoch": 30, \
                    "momentum": 0.9,  "wd": 1e-4, "batch_size": 256, \
                    "check_each_epoch": 10, "num_epoch": 120, \
                    "metric": 'acc', "fixed_param_names": None
                   })

    log_file = get_file_name_by_time(prefix="./logs/", ext=".log")

    dataset_path = "/home/zhangyasen/dataset/ILSVRC/"

    train_lst_file = os.path.join(dataset_path, "data/train.lst")
    train_rec_file = os.path.join(dataset_path, "data/train_480_q90.1000.rec")

    val_lst_file = os.path.join(dataset_path, "data/val.lst")
    val_rec_file = os.path.join(dataset_path, "data/val_256_q90.1000.rec")

    train_dataiter, val_dataiter = get_dataiter(train_rec_file, val_rec_file)
    devs = [mx.gpu(i) for i in range(4)]

    mx_train(log_file, len(open(train_lst_file).readlines()), drn_sym, \
            None, None, train_dataiter, val_dataiter, devs, "../result/drn_18_imagenet", train_params)
