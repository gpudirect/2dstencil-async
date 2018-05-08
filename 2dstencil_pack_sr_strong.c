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
#include "validate_strong.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "pack_strong.h"
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
int max_comm_size = 64*1024;
int comm_size, px, py;
int comm_rank, rank_coords[2], rank_base[2];
int left, right, bottom, top;
int threadsperblock = 512, gridsize = 15;

static inline int intlog2(int x) {
    int result = 0, temp = x;
    while (temp >>= 1) result++;
    return result;
}

static inline int intpow2(int x) {
    int result = 1, temp = x;
    result = result << temp;
    return result;
}

int exchange (MPI_Comm comm2d, int npx, int npy, long long int sizex, long long int sizey, int boundary, int iter_count, int iter_warmup)
{
    long long int i, buf_size, msg_size, msg_size_bytes, packbuf_disp, size; 
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

    long long int boundary_log, sizex_log, sizey_log, sizexy_log;
    float *compute_buf, *exchange_buf, *temp;

    size = sizex > sizey ? sizex : sizey; 
    buf_size = sizeof(float)*(sizex+2)*(sizey+2);
    msg_size = size;
    msg_size_bytes = size*sizeof(float);

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

    for (x = 0; x < sizex; x++) {
        for (y = 0; y < sizey; y++) {
            *(buf_u_h + (x+1)*(sizey+2) + (y+1))
              = 1.0;
             *(buf_v_h + (x+1)*(sizey+2) + (y+1))
              = 1.0; 
        }
    } 

    cudaMemcpy (buf_u, buf_u_h, buf_size, cudaMemcpyHostToDevice);
    cudaMemcpy (buf_v, buf_v_h, buf_size, cudaMemcpyHostToDevice);
#endif

    sizex_log = intlog2(sizex);	
    sizey_log = intlog2(sizey);	
    sizexy_log = intlog2(sizex*sizey);	
    boundary_log = intlog2(boundary);	

    copytosymbol (sizex, sizey, boundary, sizex_log, sizey_log, boundary_log, sizexy_log);

    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaStreamCreateWithFlags(&interior_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&boundary_stream, cudaStreamNonBlocking));

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
	interior_compute (compute_buf, exchange_buf, sizex, sizey, boundary, threadsperblock, gridsize, interior_stream);
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
#endif
        sreq_idx = 0;

#ifndef _FREE_PACK_
	/*pack data*/
   	boundary_pack (packbuf, exchange_buf, sizex, sizey, threadsperblock, boundary_stream);
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

        MPI_Waitall(rreq_idx, rreq, MPI_STATUS_IGNORE);
        MPI_Waitall(sreq_idx, sreq, MPI_STATUS_IGNORE);
#endif

#ifndef _FREE_PACK_
        /*unpack data*/
        boundary_unpack (exchange_buf, unpackbuf, sizex, sizey, threadsperblock, boundary_stream);
#endif

#ifndef _FREE_BOUNDARY_COMPUTE_
	/*launch boundary computation*/
        boundary_compute (compute_buf, exchange_buf, sizex, sizey, boundary, threadsperblock, gridsize, boundary_stream);
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
        fprintf(stderr, "%dx%d %8.2lf usec\n", npx, npy, (time_elapsed*1000)/iter_count);
    }

#if defined (_ENABLE_PROFILING_)
    cudaProfilerStop();
    fprintf(stderr, "stopped profiling \n");
#endif

#ifdef _VALIDATE_
    emulate_on_host (buf_u_h, buf_v_h, sizex, sizey, boundary, 1/*ghost*/, 
            comm_rank, left, right, bottom, top, (iter_count + iter_warmup), comm2d);

    cudaMemcpy (buf_u_h, buf_u, buf_size, cudaMemcpyDeviceToHost);
    cudaMemcpy (buf_v_h, buf_v, buf_size, cudaMemcpyDeviceToHost);

    validate (buf_u_h, buf_v_h, sizex, sizey, 1);
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
    int npx, npy, sizex, sizey, count; 
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
    if (getenv("COMM_SIZE") != NULL) {
        max_comm_size = atoi(getenv("COMM_SIZE"));
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
    period[0] = 1;
    period[1] = 1;
    reorder = 0;

    MPI_Cart_create (MPI_COMM_WORLD, 2, dim, period, reorder, &comm2d);
    MPI_Cart_shift(comm2d, 0,  1,  &left, &right );
    MPI_Cart_shift(comm2d, 1,  1,  &bottom, &top );
    MPI_Comm_rank(comm2d, &comm_rank);

    MPI_Cart_coords(comm2d, comm_rank, 2, rank_coords);

    MPI_Barrier(MPI_COMM_WORLD);

    fprintf(stderr, "[%d] size: %lld left: %d right: %d top: %d bottom: %d iter_count: %d iter_warmup: %d\n", comm_rank, max_size, left, right, top, bottom, iter_count, iter_warmup);

#if defined (_ENABLE_PROFILING_) 
    if (c < 3) {
        fprintf(stderr, "in profiling mode, the program takes two arguments: npx  npy \n");
        exit(-1);
    }
    npx = atoi(v[1]);
    npy = atoi(v[2]);
    if (npx <= 0 || npy <= 0) {
        fprintf(stderr, "invalid size \n");
        exit(-1);
    }
    {
#else
    int temp = intlog2(comm_size);
    npx = intpow2 (temp/2);
    npy = intpow2 (temp - (temp/2));
    for (count = comm_size; count <= max_comm_size; count*=2) 
    {
	assert (count == (npx*npy));
	sizex = max_size/npx;
        sizey = max_size/npy; 
#endif
        exchange(comm2d, npx, npy, sizex, sizey, boundary_compute_width, iter_count, iter_warmup);

        if (npy == npx) {
            npy *= 2;
        } else {
            npx *= 2;
        }
    }

    MPI_Comm_free (&comm2d);
    MPI_Finalize();

    return 0;
}
