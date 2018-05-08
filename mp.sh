#!/bin/bash

#set -x

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters, takes two : ./mp.sh PX PY"
    exit	
fi

self=$0
path=${self%/*}

if [ -z $path ]; then path='.'; fi

export PX=$1
export PY=$2
export BATCHES_INFLIGHT=4
export ITERS_PER_BATCH=10 
export ITER_COUNT=300 
export WARMUP_COUNT=300 
export MV2_ENABLE_AFFINITY=0

#export BATCHES_INFLIGHT=4
#export ITERS_PER_BATCH=10 
#export ITER_COUNT=10
#export MIN_SIZE=32
#export MAX_SIZE=32
#export MIN_SIZE=512
#export MAX_SIZE=512
#export WARMUP_COUNT=0
#export IB_CQ_DEPTH=128

export GDS_ENABLE_DEBUG
export MP_ENABLE_DEBUG
export USE_GPU_ASYNC
export USE_SINGLE_STREAM

export WAIT_FOR_KEY

#export USE_IB_HCA=mlx5_0
#export USE_GPU=0
#export CPU=0
#echo IB $USE_IB_HCA GPU $USE_GPU CPU $CPU

export CUDA_VISIBLE_DEVICES USE_IB_HCA USE_GPU USE_CPU

echo "--> ${HOSTNAME}: picking GPU:$CUDA_VISIBLE_DEVICES CPU:$USE_CPU HCA:$USE_IB_HCA"

exe=$path/2dstencil_pack_mp

exec $exe




