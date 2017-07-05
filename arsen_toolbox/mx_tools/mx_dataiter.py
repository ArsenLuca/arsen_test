# -*- coding: utf-8 -*-

import os
import mxnet as mx

def get_dataiter(train_rec_file, val_rec_file, batch_size=128, data_shape=(3, 224, 224)):
    """ get dataiter """
    mean_file = os.path.join(os.path.split(train_rec_file)[0], "imagenet_mean.bin")

    train_dataiter = mx.io.ImageRecordIter(
        path_imgrec         = train_rec_file,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        shuffle             = True,
        rand_crop           = True,
        rand_mirror         = True,
        max_random_scale    = 1.0,  # 480 with imagnet
        min_random_scale    = 0.533,  # 256.0/480.0
        max_aspect_ratio    = 0.25,
        random_h            = 36,  # 0.4*90
        random_s            = 50,  # 0.4*127
        random_l            = 50  # 0.4*127
        #mean_img            = mean_file
        )

    val_dataiter = mx.io.ImageRecordIter(
        path_imgrec         = val_rec_file,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        rand_crop           = False,
        rand_mirror         = False
        #mean_img            = mean_file
        )
    
    return (train_dataiter, val_dataiter)
