#!/usr/bin/env sh
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

# you'd better change setting with your own --data-dir, --depth, --batch-size, --gpus.
# train cifar10
#python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 164 --batch-size 128 --num-examples 50000 --gpus=0
#python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 164 --batch-size 128 --num-examples 50000 --gpus=2,3,4,5,6,7

## train resnet-50 
nohup python -u train_drn.py --network dilated_resnet --num_layer 18 --dataset imagenet > train_drn.output 2>&1 &