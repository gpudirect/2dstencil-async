#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters, takes two : ./ib.sh PX PY"
    exit
fi

self=$0
path=${self%/*}

if [ -z $path ]; then path='.'; fi

export PX=$1
export PY=$2
export BATCHES_INFLIGHT=4
export ITERS_PER_BATCH=10 
export ITER_COUNT=200 
export WARMUP_COUNT=300 
export MV2_ENABLE_AFFINITY=0

export CPU=0
export USE_IB_HCA=mlx5_0
export USE_GPU=0
export CPU=1

exec $path/2dstencil_pack_ib
