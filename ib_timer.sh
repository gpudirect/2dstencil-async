#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters, takes three : ./ib_timer.sh PX PY DIM"
    exit
fi

export PX=$1
export PY=$2
export BATCHES_INFLIGHT=4
export ITERS_PER_BATCH=10 
export ITER_COUNT=200 
export WARMUP_COUNT=300 
export MV2_ENABLE_AFFINITY=0

export CPU=0
if [ $MV2_COMM_WORLD_LOCAL_RANK -eq 0 ]; then 
    export USE_IB_HCA=mlx5_0
    export USE_GPU=0
    export CPU=1
    echo IB $USE_IB_HCA GPU $USE_GPU
else
    export USE_IB_HCA=mlx5_1
    export USE_GPU=1
    export CPU=11
    echo IB $USE_IB_HCA GPU $USE_GPU
fi

numactl --physcpubind=${CPU} -l ./2dstencil_pack_ib $3
