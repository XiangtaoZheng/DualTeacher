#!/usr/bin/env bash
set -x

TYPE=$1
FOLD=$2
PERCENT=$3
GPUS=$4
PORT=${PORT:-29500}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py configs/baseline/faster_rcnn_r101_caffe_fpn_${TYPE}_full_180k.py --launcher pytorch \
    --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5}

