/****
 * Copyright (c) 2011-2014, NVIDIA Corporation.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA Corporation nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 ****/

#include <stdio.h>
#include <assert.h>
#include "mpi.h"
#include "validate.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "pack.h"
#include <string.h>

#define CUDA_CHECK(stmt)                                \
do {                                                    \
    cudaError_t result = (stmt);                        \
    if (cudaSuccess != result) {                        \
        fprintf(stderr, "[%s:%d] cuda failed with %s \n",   \
         __FILE__, __LINE__,cudaGetErrorString(result));\
        exit(-1);                                       \
    }                                                   \
    assert(cudaSuccess == result);                      \
} while (0)

#define CU_CHECK(stmt)                                  \
do {                                                    \
    CUresult result = (stmt);                           \
    if (CUDA_SUCCESS != result) {                       \
        fprintf(stderr, "[%s:%d] cu failed with %d \n", \
         __FILE__, __LINE__,  result);                  \
        exit(-1);                                       \
    }                                                   \
    assert(CUDA_SUCCESS == result);                     \
} while (0)

/*    x         
 *    ^      
 *    -    
 *    -  
 *    -
 *  0 --------> y
 * */

long long int max_size = 8*1024, default_boundary = 2;
int comm_size, px, py;
int comm_rank, rank_coords[2], rank_base[2];
int left, right, bottom, top;
int threadsperblock = 512, gridsize = 15;

static inline int intlog2(int x) {
    int result = 0, temp = x;
    while (temp >>= 1) result++;
    return result;
}

int exchange (MPI_Comm comm2d, long long int size, int boundary, int iter_count, int iter_warmup)
{
    long long int i, buf_size, msg_size, msg_size_bytes, packbuf_disp; 
    int neighbors = 4;
    float *buf_u = NULL, *buf_v = NULL, *packbuf = NULL, *unpackbuf = NULL;
#ifdef _VALIDATE_
    int x, y;
    float *buf_u_h = NULL, *buf_v_h = NULL;
#endif

    int rreq_idx, sreq_idx;
    MPI_Request *sreq = NULL, *rreq = NULL;
    cudaEvent_t start_event, stop_event;
    float time_elapsed;
    cudaStream_t interior_stream, boundary_stream; 

    long long int boundary_log, size_log, size2_log;
    float *compute_buf, *exchange_buf, *temp;

    buf_size = sizeof(float)*(size+2)*(size+2);
    msg_size = size;
    msg_size_bytes = msg_size*sizeof(float);

    /*allocating requests*/	
    sreq = (MPI_Request *) malloc(neighbors*sizeof(MPI_Request));
    rreq = (MPI_Request *) malloc(neighbors*sizeof(MPI_Request));

    CUDA_CHECK(cudaMalloc((void **)&buf_u, buf_size));
    CUDA_CHECK(cudaMalloc((void **)&buf_v, buf_size));
    CUDA_CHECK(cudaMalloc((void **)&packbuf, msg_size_bytes*neighbors));
    CUDA_CHECK(cudaMalloc((void **)&unpackbuf, msg_size_bytes*neighbors));

    
    CUDA_CHECK(cudaMemset(packbuf, 0, msg_size_bytes*neighbors));
    CUDA_CHECK(cudaMemset(unpackbuf, 0, msg_size_bytes*neighbors));

#ifdef _VALIDATE_
    buf_u_h = malloc (buf_size); 
    buf_v_h = malloc (buf_size);
    memset(buf_u_h, 0, buf_size);
    memset(buf_v_h, 0, buf_size);

    for (x = 0; x < size; x++) {
        for (y = 0; y < size; y++) {
            *(buf_u_h + (x+1)*(size+2) + (y+1))
              = 1.0;
             *(buf_v_h + (x+1)*(size+2) + (y+1))
              = 1.0; 
        }
    } 

    cudaMemcpy (buf_u, buf_u_h, buf_size, cudaMemcpyHostToDevice);
    cudaMemcpy (buf_v, buf_v_h, buf_size, cudaMemcpyHostToDevice);
#endif

    size_log = intlog2(size);	
    size2_log = intlog2(size*size);	
    boundary_log = intlog2(boundary); 

    copytosymbol (size, size_log, size2_log, boundary, boundary_log);

    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaStreamCreate(&interior_stream));
    CUDA_CHECK(cudaStreamCreate(&boundary_stream));
    //CUDA_CHECK(cudaStreamCreateWithFlags(&interior_stream, cudaStreamNonBlocking));
    //CUDA_CHECK(cudaStreamCreateWithFlags(&boundary_stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaDeviceSynchronize());

    compute_buf = buf_u; 
    exchange_buf = buf_v;

    for (i = 0; i < (iter_count + iter_warmup); i++) {
        if(i == iter_warmup) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start_event, 0));
#if defined (_ENABLE_PROFILING_)
	    fprintf(stderr, "starting profiling \n");
	    cudaProfilerStart();
#endif
        }

	rreq_idx = 0;
        sreq_idx = 0;

#ifndef _FREE_INTERIOR_COMPUTE_
	/*launch interior compute*/
	interior_compute (compute_buf, exchange_buf, size, boundary, threadsperblock, gridsize, interior_stream);
#endif

	/*exchange boundary data*/
        /*post all receives*/
	packbuf_disp = 0;

#ifndef _FREE_NETWORK_
        /*y dim*/
        if (left != -1) { 
            MPI_Irecv((void *)(unpackbuf + packbuf_disp), 
	    	msg_size, MPI_FLOAT, left, left, comm2d, 
                    &rreq[rreq_idx]);
            rreq_idx++;
        }
	packbuf_disp += msg_size;
        if (right != -1) { 
            MPI_Irecv((void *)(unpackbuf + packbuf_disp),
                    msg_size, MPI_FLOAT, right, right, comm2d, 
                    &rreq[rreq_idx]);
            rreq_idx++;
        }
        packbuf_disp += msg_size;
        /*x dim*/
        if (bottom != -1) { 
            MPI_Irecv((void *)(unpackbuf + packbuf_disp),
                msg_size, MPI_FLOAT, bottom, bottom, comm2d, 
                &rreq[rreq_idx]);
            rreq_idx++;
        }
        packbuf_disp += msg_size;
        if (top != -1) { 
            MPI_Irecv((void *)(unpackbuf + packbuf_disp),
                msg_size, MPI_FLOAT, top, top, comm2d, 
                &rreq[rreq_idx]);
            rreq_idx++;
        }
        packbuf_disp += msg_size;
#endif
        sreq_idx = 0;

#ifndef _FREE_PACK_
	/*pack data*/
   	boundary_pack (packbuf, exchange_buf, size, threadsperblock, boundary_stream);
#endif

	CUDA_CHECK(cudaStreamSynchronize(boundary_stream));

        /*post all sends*/ 
        packbuf_disp = 0;

#ifndef _FREE_NETWORK_
        /*y dim*/
        if (left != -1) {
            MPI_Isend((void *)(packbuf + packbuf_disp),
                    msg_size, MPI_FLOAT, left, comm_rank, comm2d, 
                    &sreq[sreq_idx]);
            sreq_idx++;
        }
        packbuf_disp += msg_size;
        if (right != -1) {
            MPI_Isend((void *)(packbuf + packbuf_disp),
                    msg_size, MPI_FLOAT, right, comm_rank, comm2d, 
                    &sreq[sreq_idx]);
            sreq_idx++;
        }
        packbuf_disp += msg_size;
        /*x dim*/
        if (bottom != -1) {
            MPI_Isend((void *)(packbuf + packbuf_disp),
                    msg_size, MPI_FLOAT, bottom, comm_rank, comm2d, 
                    &sreq[sreq_idx]);
            sreq_idx++;
        }
        packbuf_disp += msg_size;
        if (top != -1) {
            MPI_Isend((void *)(packbuf + packbuf_disp), 
                msg_size, MPI_FLOAT, top, comm_rank, comm2d, 
                &sreq[sreq_idx]);
            sreq_idx++;
        }
        packbuf_disp += msg_size;

        MPI_Waitall(rreq_idx, rreq, MPI_STATUS_IGNORE);
        MPI_Waitall(sreq_idx, sreq, MPI_STATUS_IGNORE);
#endif

#ifndef _FREE_PACK_
        /*unpack data*/
        boundary_unpack (exchange_buf, unpackbuf, size, threadsperblock, boundary_stream);
#endif

#ifndef _FREE_BOUNDARY_COMPUTE_
	/*launch boundary computation*/
        boundary_compute (compute_buf, exchange_buf, size, boundary, threadsperblock, gridsize, boundary_stream);
#endif

	CUDA_CHECK(cudaDeviceSynchronize());

	/*intercahnge the compute and communication buffers*/
        temp = exchange_buf; 
	exchange_buf = compute_buf; 
	compute_buf = temp;
    }

    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&time_elapsed, start_event, stop_event));
    if (comm_rank == 0) {
        fprintf(stderr, "%3lldx%3lld %8.2lf usec\n", size, size, (time_elapsed*1000)/iter_count);
    }

#if defined (_ENABLE_PROFILING_)
    cudaProfilerStop();
    fprintf(stderr, "stopped profiling \n");
#endif

#ifdef _VALIDATE_
    emulate_on_host (buf_u_h, buf_v_h, size, boundary, 1/*ghost*/, 
            comm_rank, left, right, bottom, top, (iter_count + iter_warmup), comm2d);

    cudaMemcpy (buf_u_h, buf_u, buf_size, cudaMemcpyDeviceToHost);
    cudaMemcpy (buf_v_h, buf_v, buf_size, cudaMemcpyDeviceToHost);

    validate (buf_u_h, buf_v_h, size, 1);
#endif

    CUDA_CHECK(cudaFree(buf_u));
    CUDA_CHECK(cudaFree(buf_v));
    CUDA_CHECK(cudaFree(packbuf));
    CUDA_CHECK(cudaFree(unpackbuf));

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaStreamDestroy(interior_stream));
    CUDA_CHECK(cudaStreamDestroy(boundary_stream));

    free(sreq);
    free(rreq);

    return 0;
}

int main (int c, char *v[])
{
    int iter_count, iter_warmup;
    int dim[2], period[2];
    int reorder, boundary_compute_width;
    MPI_Comm comm2d;

    px=4;
    py=4;
#if defined (_VALIDATE_)
    iter_count=30;
    iter_warmup=10;
#else
    iter_count=200;
    iter_warmup=20;
#endif
    boundary_compute_width = default_boundary;

    if (getenv("PX") != NULL) {
        px = atoi(getenv("PX"));
    }
    if (getenv("PY") != NULL) {
        py = atoi(getenv("PY"));
    }
    if (getenv("SIZE") != NULL) {
        max_size = atoi(getenv("SIZE"));
    }
    if (getenv("ITER_COUNT") != NULL) {
        iter_count = atoi(getenv("ITER_COUNT"));
    }
    if (getenv("WARMUP_COUNT") != NULL) {
        iter_warmup = atoi(getenv("WARMUP_COUNT"));
    }
    if (getenv("BOUNDARY_COMPUTE_WIDTH") != NULL) {
        boundary_compute_width = atoi(getenv("BOUNDARY_COMPUTE_WIDTH"));
    }
    if (getenv("CUDA_THREADS_PER_BLOCK") != NULL) {
        threadsperblock = atoi(getenv("CUDA_THREADS_PER_BLOCK"));
    }
    if (getenv("CUDA_GRID_SIZE") != NULL) {
        gridsize = atoi(getenv("CUDA_GRID_SIZE"));
    }

    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0) {
        fprintf(stderr, "no CUDA devices found \n");
        exit(-1);
    }

    MPI_Init(&c, &v);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);


    if (comm_size != px*py) {
        fprintf(stderr, "comm size and number of processes do not match \n");
        fprintf(stderr, "comm_size = %d, px = %d, py = %d\n",
                    comm_size, px, py);
        exit(-1);
    }

    int local_rank = 0;
    if (getenv("MV2_COMM_WORLD_LOCAL_RANK") != NULL) {
        local_rank = atoi(getenv("MV2_COMM_WORLD_LOCAL_RANK"));
    }
    cudaSetDevice(local_rank%dev_count);

    /*create the stencil communicator*/
    dim[0]    = py;
    dim[1]    = px;
    period[0] = 0;
    period[1] = 0;
    reorder = 0;

    MPI_Cart_create (MPI_COMM_WORLD, 2, dim, period, reorder, &comm2d);
    MPI_Cart_shift(comm2d, 0,  1,  &left, &right );
    MPI_Cart_shift(comm2d, 1,  1,  &bottom, &top );
    MPI_Comm_rank(comm2d, &comm_rank);

    MPI_Cart_coords(comm2d, comm_rank, 2, rank_coords);

    fprintf(stderr, "px: %d py: %d left: %d right: %d top: %d bottom: %d \n", px, py, left, right, top, bottom);

    MPI_Barrier(MPI_COMM_WORLD);

    long long int i;
#if defined (_ENABLE_PROFILING_) 
    if (c < 2) {
        fprintf(stderr, "in profiling mode, the program takes one argument: stencil dimension (eg: 512, 16384) \n");
        exit(-1);
    }
    i = atoi(v[1]);
    if (i <= 0) {
        fprintf(stderr, "invalid size \n");
        exit(-1);
    }
#else
    for (i=4*boundary_compute_width; i<=4*boundary_compute_width/*max_size*/; i*=2) 
#endif
    {	
        exchange(comm2d, i, boundary_compute_width, iter_count, iter_warmup);
    }

    MPI_Comm_free (&comm2d);
    MPI_Finalize();

    return 0;
}
