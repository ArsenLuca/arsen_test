# -*- coding: utf-8 -*-

import os
import math
import time
import numpy as np
import cv2

from ..file_tools.file_scan import list_file

def get_img_list(img_folder):
    image_list = list_file(img_folder, recursive=True, exts=[".jpg", ".jpeg"])
    return [image_info[1] for image_info in image_list]

def get_image(filename):
    """ load image (opencv) from file """
    img = cv2.imread(filename)  # read image in b,g,r order
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # change to r,g,b order
    img = cv2.resize(img, (224, 224))  # resize to 224*224 to fit model
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)  # change to (channel, height, width)
    img = img[np.newaxis, :]  # extend to (example, channel, heigth, width)
    return img

def mx_load_module(batch_size, checkpoint_name, epoch, context):
    """ load mx model(checkpoint) into module """
    sym, arg_params, aux_params = mx.model.load_checkpoint(checkpoint_name, epoch)
    mod = mx.mod.Module(symbol=sym, label_names=None, context=context)
    mod.bind(for_training=False, data_shapes=[('data', (batch_size, 3, 224, 224))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    return mod

def mx_imagely_predict(img_list, mod, batch_size, val_label):
    """ mx predict img_list """

    acc = 0.0
    total = 0.0
    acc_top5 = 0.0

    for i in range(0, int(math.ceil(1.0*len(img_list)/batch_size))):
        tic = time.time()

        end_id = min((i+1)*batch_size, len(img_list))
        idx = range(i*batch_size, end_id)

        img = np.concatenate([get_image(img_list[j]) for j in idx])
        mod.forward(Batch([mx.nd.array(img)]))
        prob = mod.get_outputs()[0].asnumpy()

        pred = np.argsort(prob, axis=1)
        labels = np.array([val_label[j] for j in idx])

        for pred_label, label in zip(pred_labels, labels):
            top5_labels = pred_label[-5:]
            if label in top5_labels:
                acc_top5 += 1.0 
            if label == pred_label[-1]:
                acc += 1.0 

        total += len(idx)
        print('batch %d, time %f sec'%(i, time.time() - tic))

    print('Top 1 accuracy: %f'%(acc/total))
    print('Top 5 accuracy: %f'%(acc_top5/total))

    return {"top1": val_dataiter, "top5": acc_top5/total}

def mx_iterly_predict(val_dataiter, mod, batch_size):
    """ mx predict dataiter """

    acc = 0.0
    total = 0.0
    acc_top5 = 0.0

    for preds, i_batch, batch in mod.iter_predict(val_dataiter):
        tic = time.time()

        print("\tpredict batch {0}".format(i_batch))
        pred_labels = preds[0].asnumpy().argsort(axis=1)
        labels = batch.label[0].asnumpy().astype('int32')[0:pred_labels.shape[0]]

        for pred_label, label in zip(pred_labels, labels):
            top5_labels = pred_label[-5:]
            top1_label = pred_label[-1]
            if label in top5_labels:
                acc_top5 += 1.0 
            if label == top1_label:
                acc += 1.0 
            
        total += preds[0].shape[0]
        print('batch %d, time %f sec'%(i_batch, time.time() - tic))

    print('Top 1 accuracy: %f'%(acc/total))
    print('Top 5 accuracy: %f'%(acc_top5/total))

    return {"top1": val_dataiter, "top5": acc_top5/total}

def mx_imagely_load_predict(img_folder, checkpoint_name, epoch, batch_size, val_label, context=mx.gpu()):
    """ load images and predict """
    mod = mx_load_module(batch_size, checkpoint_name, epoch, context)
    return mx_imagely_predict(get_img_list(img_folder), mod, batch_size, val_label)

def mx_iterly_load_predict(val_dataiter, checkpoint_name, epoch, batch_size, context=mx.gpu()):
    """ load dataiter and predict """
    mod = mx_load_module(batch_size, checkpoint_name, epoch, context)
    return mx_iterly_predict(val_dataiter, mod, batch_size)
