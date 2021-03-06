# -*- coding: utf-8 -*-

import mxnet as mx
import numpy as np
from ..basic.log import set_logger

def mx_train(log_file_name, train_num, sym, arg_params, aux_params, 
            train_dataiter, val_dataiter, devs, save_model_prefix, train_params):
    ''' Train CNN Network '''

    lr_factor_epoch = train_params.lr_factor_epoch
    num_epoch = train_params.num_epoch
    batch_size = train_params.batch_size
    metric = train_params.metric
    frequent = train_params.frequent
    fixed_param_names = train_params.fixed_param_names
    ### Set Module ###

    # set logger
    logger = set_logger('', log_file_name)
    # module
    mod = mx.mod.Module(symbol=sym, context=devs, logger=logger,
                        fixed_param_names=fixed_param_names)
    # bind
    mod.bind(data_shapes=train_dataiter.provide_data, label_shapes=train_dataiter.provide_label)
    # initialization
    mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
    # set params
    if arg_params != None:  # whether using pretrained model or not
        logger.info("Using Pretrained model params")
        mod.set_params(arg_params, aux_params, allow_missing=True)

    # set checkpoints
    check_each_epoch = min(train_params.check_each_epoch, num_epoch)
    checkpoint = mx.callback.do_checkpoint(save_model_prefix, check_each_epoch)

    ### Set Optimizer ###

    # how to change lr
    base_lr = train_params.base_lr
    lr_factor = train_params.lr_factor
    momentum = train_params.momentum
    batch_num = mx.nd.np.ceil(float(train_num) / batch_size)  # flipped

    # weight of regularization
    wd = train_params.wd

    if isinstance(lr_factor_epoch, int):
        lr_scheduler = mx.lr_scheduler.FactorScheduler(
            step=max(int(batch_num * lr_factor_epoch), 1), factor=lr_factor)
    elif isinstance(lr_factor_epoch, np.ndarray):
        lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(
            step=list(batch_num * lr_factor_epoch), factor=lr_factor)
    else:
        raise Exception("Unsupported lr_factor_epoch Type")

    optimizer = mx.optimizer.Optimizer.create_optimizer('sgd',
                                                        learning_rate=base_lr,
                                                        rescale_grad=1.0 / batch_size,
                                                        momentum=momentum,
                                                        wd=wd,
                                                        lr_scheduler=lr_scheduler)

    ### Training the module ###
    mod.fit(train_dataiter, val_dataiter,
            num_epoch=num_epoch,
            batch_end_callback=mx.callback.Speedometer(batch_size, frequent),
            kvstore='device',
            optimizer=optimizer,
            epoch_end_callback=checkpoint,
            eval_metric=metric)
