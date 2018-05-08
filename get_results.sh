#!/bin/bash


make clean
make 2dstencil_pack_ib 2dstencil_pack_mp 2dstencil_pack_ib_strong 2dstencil_pack_mp_strong

echo "2stencil without gpudirect sync"

mpiexec.hydra -envall -n 2 -hosts drossetti-ivy2,drossetti-ivy3 ./ib.sh 1 2

echo "2stencil with gpudirect sync"

mpiexec.hydra -envall -n 2 -hosts drossetti-ivy2,drossetti-ivy3 ./mp.sh 1 2

echo "2stencil strong scaling without gpudirect sync"

mpiexec.hydra -envall -n 2 -hosts drossetti-ivy2,drossetti-ivy3 ./ib_strong.sh 1 2

echo "2stencil strong scaling with gpudirect sync"

mpiexec.hydra -envall -n 2 -hosts drossetti-ivy2,drossetti-ivy3 ./mp_strong.sh 1 2


