# Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include ../common.mk
# install path
PREFIX ?= $(HOME)/gdasync

CUDA_LIB :=
CUDA_INC :=

# # add standard CUDA dir
CUDA      ?= /usr/local/cuda
CUDA_LIB += $(CU_LDFLAGS) -L$(CUDA)/lib64 -L$(CUDA)/lib -L/usr/lib64/nvidia -L/usr/lib/nvidia
CUDA_INC += $(CU_CPPFLAGS) -I$(CUDA)/include

# # GPUDirect Async library
GDSYNC      = $(PREFIX)
GDSYNC_INC  = -I $(GDSYNC)/include
GDSYNC_LIB  = -L $(GDSYNC)/lib

# MP library
MP     ?= $(PREFIX)
MP_INC  = -I $(MP)/include
MP_LIB  = -L $(MP)/lib

# GPUDirect RDMA Copy library
GDRCOPY     ?= $(PREFIX)
GDRCOPY_INC  = -I $(GDRCOPY)/include
GDRCOPY_LIB  = -L $(GDRCOPY)/lib

# MPI is used in some tests
MPI_HOME ?= /ivylogin/home/spotluri/MPI/mvapich2-2.0-cuda-gnu-install
MPI_INC   = -I $(MPI_HOME)/include
MPI_LIB   = -L $(MPI_HOME)/lib64

CXX:=g++
CC:=mpicc
NVCC:=nvcc
LD:=mpic++
COMMON_CFLAGS:=-O2

NVCC_ARCHS=-gencode arch=compute_35,code=sm_35
NVCC_ARCHS+=-gencode arch=compute_50,code=compute_50
NVCC_ARCHS+=-gencode arch=compute_60,code=compute_60


EXTRA_FLAGS=-D_USE_NONBLOCKING_STREAMS_ -D_USE_STREAM_PRIORITY_ -D_NOWAIT_ON_SENDS_ -D_INTERIOR_FIRST_ -D_POLL_DWORD_LIST_ 
# to enable validate
#EXTRA_FLAGS+=-D_VALIDATE_
PROF_FLAGS= #-DUSE_PROF
CPPFLAGS=-I.. $(MP_INC) $(MPI_INC) $(CUDA_INC) $(EXTRA_FLAGS) $(PROF_FLAGS)

CFLAGS=$(COMMON_CFLAGS) -Wall
CXXFLAGS=$(COMMON_CFLAGS) -Wall
NVCCFLAGS=$(COMMON_CFLAGS) $(NVCC_ARCHS)

MPI_LDFLAGS=$(MPI_LIB)
CUDA_LDFLAGS=$(CUDA_LIB) -lcudart -L/usr/lib64 -lcuda
GDRCOPY_LDFLAGS=$(GDRCOPY_LIB) -lgdrapi
OFA_LDFLAGS=$(OFA_LIB) -libverbs
MP_LDFLAGS = $(MP_LIB) -lmp $(GDSYNC_LIB) -lgdsync

#ifdef USE_PROF
#TOOLS_LDFLAGS=$(TOOLS_LIB) -lgdstools
TOOLS_LDFLAGS=
LDFLAGS=-L.. $(TOOLS_LDFLAGS) $(MPI_LDFLAGS) $(MP_LDFLAGS) $(OFA_LDFLAGS) $(GDRCOPY_LDFLAGS) $(CUDA_LDFLAGS) -ldl -lstdc++

#-D_ENABLE_PROFILING_ -D_ENABLE_DRPROF_ -D_ENABLE_DRPROF_ISEND_ 
#-D_USE_STREAM_PRIORITY_
#-D_POLL_DWORD_LIST_
#-D_USE_POST_SEND_ONE_ 
#-D_REPEAT_TIMED_LOOP_ 
#-D_ENABLE_PROFILING_ -D_ENABLE_DRPROF_ 
#-D_FREE_BOUNDARY_COMPUTE_ -D_FREE_INTERIOR_COMPUTE_ -D_FREE_PACK_
#-D_FREE_BOUNDARY_COMPUTE_ -D_FREE_INTERIOR_COMPUTE_ -D_FREE_PACK_ 
#-D_ENABLE_PROFILING_ -D_ENABLE_DRPROF_ 

#-D_ENABLE_PROFILING_
#-D_FREE_SYNC_
#-D_FREE_NETWORK_ 
#-D_FREE_BOUNDARY_COMPUTE_ -D_FREE_INTERIOR_COMPUTE_ -D_FREE_PACK_ 
#-D_FREE_SYNC_
#-D_ENABLE_DRPROF_ -DUSE_PROF
#-D_FREE_BOUNDARY_COMPUTE_
#-D_FREE_NETWORK_ 
#-D_FREE_INTERIOR_COMPUTE_
#-D_FREE_PACK_

CSRCS=validate.c validate_strong.c ib.c 2dstencil_pack_sr.c 2dstencil_pack_sr.c 2dstencil_pack_ib.c 2dstencil_pack_ib_strong.c 2dstencil_pack_mp.c 2dstencil_pack_mp_strong.c 2dstencil_pack_mp_strong_3streams.c 2dstencil_pack_mp_strong_3streams_uevent.c 2dstencil_sgl_mp.c
CCSRCS=
CUSRCS=pack.cu pack_strong.cu
# gpu.cu 

OBJS=$(CSRCS:%.c=%.o) $(CCSRCS:%.cpp=%.o) $(CUSRCS:%.cu=%.o)
#EXES=$(OBJS:%.o=%)
EXES=2dstencil_pack_sr 2dstencil_pack_ib 2dstencil_pack_mp 2dstencil_sgl_mp 2dstencil_pack_sr_strong 2dstencil_pack_ib_strong 2dstencil_pack_mp_strong 2dstencil_pack_mp_strong_3streams 
# 2dstencil_pack_mp_strong_3streams_uevent

# Commands
all: $(EXES)

%: %.o
	$(LD) -o $@ $^ $(LDFLAGS)

2dstencil_pack_mp: pack.o 2dstencil_pack_mp.o

2dstencil_pack_sr: pack.o 2dstencil_pack_sr.o validate.o

2dstencil_pack_sr_strong: pack_strong.o 2dstencil_pack_sr_strong.o validate_strong.o

2dstencil_pack_ib: pack.o 2dstencil_pack_ib.o ib.o

2dstencil_pack_ib_strong: pack_strong.o 2dstencil_pack_ib_strong.o ib.o

2dstencil_pack_mp: pack.o 2dstencil_pack_mp.o

2dstencil_pack_mp_strong: pack_strong.o 2dstencil_pack_mp_strong.o

2dstencil_pack_mp_strong_3streams: pack_strong.o 2dstencil_pack_mp_strong_3streams.o

2dstencil_pack_mp_strong_3streams_uevent: pack_strong.o 2dstencil_pack_mp_strong_3streams_uevent.o

2dstencil_sgl_mp: pack.o 2dstencil_sgl_mp.o

clean:
	rm -rf $(EXES) *.o 

.PHONY: clean all depend
#include ../depend.mk
