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
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include "validate.h"
#include <mp.h>
#include "pack.h"
#include <string.h>
#include "prof.h"
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>


/*    x         
 *    ^      
 *    -    
 *    -  
 *    -
 *  0 --------> y
 * */

#define CUDA_CHECK(stmt)                                \
do {                                                    \
    cudaError_t result = (stmt);                        \
    if (cudaSuccess != result) {                        \
        fprintf(stderr, "[%d][%d][%s:%d] cuda failed with %s \n",   \
                comm_rank, getpid(), __FILE__, __LINE__,cudaGetErrorString(result)); \
        exit(-1);                                       \
    }                                                   \
    assert(cudaSuccess == result);                      \
} while (0)

struct prof prof;
struct prof prof_isend_prepare;
int prof_start = 0;
int prof_idx = 0;

int tracking_events = 0;
int wait_for_key = 0;
long long int min_size = 8, max_size = 8*1024, default_boundary = 2;
int comm_size, px, py;
int comm_rank = 0, rank_coords[2], rank_base[2];
int left, right, bottom, top, peers[4];
int peer_count, req_per_step; 
int sreq_max_inflight, rreq_max_inflight, steps_per_batch = 4, batches_inflight = 4, buffering = 2, prepost_depth_global = 4;
int threadsperblock = 512, gridsize = 15;
int period_value = 1, use_async = 1, use_single_stream = 0, force_stream_sync = 0;
int reps_count = 10;
static inline int intlog2(int x) {
    int result = 0, temp = x;
    while (temp >>= 1) result++;
    return result;
}
int use_gpu_comm_buffers = 0;
int use_mpi=0;

int exchange (MPI_Comm comm2d, long long int size, int boundary, int iter_count, int iter_warmup)
{
    long long int i; 
    int neighbors = 4;

    /*application and pack buffers*/
    long long int buf_size, msg_size, msg_size_bytes, packbuf_disp; 
    float *buf_u = NULL, *buf_v = NULL, **packbuf, **unpackbuf;
    float *compute_buf, *exchange_buf, *temp;

    /*counters to track requests*/
    int rreq_idx, sreq_idx, wait_idx, complete_idx, rreq_inflight;
    int sreq_inflight, complete_sync_inflight, complete_sync_idx, complete_sync_wait;
    int complete_sreq_idx, complete_rreq_idx, prepost_depth;
  
    /*mp specific objects*/
    mp_reg_t *reg_pack, *reg_unpack; 
    mp_request_t *sreq = NULL, *rreq = NULL;

    /*events for tracking and timing*/
    cudaEvent_t start_event, stop_event;
    cudaEvent_t *sendrecv_sync_event, *interior_sync_event;
    long int time_start, time_stop, time_prepost;
    float time_elapsed; 

    /*streams*/
    cudaStream_t interior_stream;
    cudaStream_t boundary_sendrecv_stream;

    /*hold constants*/
    long long int boundary_log, size_log, size2_log;

    /*variables used in validation*/
#ifdef _VALIDATE_
    int x, y;
    float *buf_u_h = NULL, *buf_v_h = NULL;
#endif

    MPI_Request * mpi_sreq, * mpi_rreq;  
    MPI_Status * mpi_sstatus, * mpi_rstatus;

    buf_size = sizeof(float)*(size+2)*(size+2);
    msg_size = size;
    msg_size_bytes = msg_size*sizeof(float);

    /*allocating requests*/
    if (use_async) { 	
        sreq_max_inflight = req_per_step*steps_per_batch*batches_inflight; 
        rreq_max_inflight = req_per_step*(steps_per_batch*batches_inflight + prepost_depth_global); 
    } else {
        sreq_max_inflight = req_per_step;
        rreq_max_inflight = req_per_step*prepost_depth_global;
    }

    if(use_mpi == 0)
    {
        sreq = (mp_request_t *) malloc(sreq_max_inflight*sizeof(mp_request_t));
        rreq = (mp_request_t *) malloc(rreq_max_inflight*sizeof(mp_request_t));         
    }
    else
    {
        mpi_sreq = (MPI_Request *) malloc(sreq_max_inflight*sizeof(MPI_Request));
        mpi_rreq = (MPI_Request *) malloc(rreq_max_inflight*sizeof(MPI_Request));
        mpi_sstatus = (MPI_Status *) malloc(sreq_max_inflight*sizeof(MPI_Status));
        mpi_rstatus = (MPI_Status *) malloc(rreq_max_inflight*sizeof(MPI_Status));
    }

    sendrecv_sync_event = (cudaEvent_t *) malloc (batches_inflight*steps_per_batch*sizeof(cudaEvent_t));
    interior_sync_event = (cudaEvent_t *) malloc (batches_inflight*steps_per_batch*sizeof(cudaEvent_t));

    packbuf = (float **) malloc (sizeof(float *)*buffering); 
    unpackbuf = (float **) malloc (sizeof(float *)*buffering); 
    reg_pack = (mp_reg_t *) malloc (sizeof(mp_reg_t)*buffering); 
    reg_unpack = (mp_reg_t *) malloc (sizeof(mp_reg_t)*buffering); 
    
    if(use_gpu_comm_buffers == 0)
    {
        CUDA_CHECK(cudaMallocHost((void **)&buf_u, buf_size));
        CUDA_CHECK(cudaMallocHost((void **)&buf_v, buf_size));
        for (i=0; i<buffering; i++) { 
            CUDA_CHECK(cudaMallocHost((void **)&packbuf[i], msg_size_bytes*neighbors));
            if(use_mpi == 0)
                mp_register(packbuf[i], msg_size_bytes*neighbors, &reg_pack[i]); 

            CUDA_CHECK(cudaMallocHost((void **)&unpackbuf[i], msg_size_bytes*neighbors));
    
            if(use_mpi == 0)
                mp_register(unpackbuf[i], msg_size_bytes*neighbors, &reg_unpack[i]); 
        }
    }
    else
    {
        CUDA_CHECK(cudaMalloc((void **)&buf_u, buf_size));
        CUDA_CHECK(cudaMalloc((void **)&buf_v, buf_size));
        for (i=0; i<buffering; i++) { 
            CUDA_CHECK(cudaMalloc((void **)&packbuf[i], msg_size_bytes*neighbors));
            CUDA_CHECK(cudaMemset(packbuf[i], 0, msg_size_bytes*neighbors));
            if(use_mpi == 0)
                mp_register(packbuf[i], msg_size_bytes*neighbors, &reg_pack[i]); 

            CUDA_CHECK(cudaMalloc((void **)&unpackbuf[i], msg_size_bytes*neighbors));
            CUDA_CHECK(cudaMemset(unpackbuf[i], 0, msg_size_bytes*neighbors));
        
            if(use_mpi == 0)
                mp_register(unpackbuf[i], msg_size_bytes*neighbors, &reg_unpack[i]); 
        }
    }

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

    if(use_gpu_comm_buffers == 0)
    {
        memcpy(buf_u, buf_u_h, buf_size);
        memcpy(buf_v, buf_v_h, buf_size);
    }
    else
    {
        CUDA_CHECK(cudaMemcpy(buf_u, buf_u_h, buf_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf_v, buf_v_h, buf_size, cudaMemcpyHostToDevice));
    }

#endif

    size_log = intlog2(size);	
    size2_log = intlog2(size*size);	
    boundary_log = intlog2(boundary); 

    copytosymbol (size, size_log, size2_log, boundary, boundary_log);

    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    for (i=0; i<batches_inflight*steps_per_batch; i++) { 
    	CUDA_CHECK(cudaEventCreateWithFlags(&sendrecv_sync_event[i], cudaEventDisableTiming));
    	CUDA_CHECK(cudaEventCreateWithFlags(&interior_sync_event[i], cudaEventDisableTiming));
    }

    int least, greatest;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
#ifndef _USE_NONBLOCKING_STREAMS_
    CUDA_CHECK(cudaStreamCreate(&interior_stream));
    CUDA_CHECK(cudaStreamCreate(&boundary_sendrecv_stream));
#else
#ifdef _USE_STREAM_PRIORITY_
    CUDA_CHECK(cudaStreamCreateWithPriority(&interior_stream, cudaStreamNonBlocking, least));
    CUDA_CHECK(cudaStreamCreateWithPriority(&boundary_sendrecv_stream, cudaStreamNonBlocking, greatest));
#else
    CUDA_CHECK(cudaStreamCreateWithFlags(&interior_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&boundary_sendrecv_stream, cudaStreamNonBlocking));
#endif
#endif

    CUDA_CHECK(cudaDeviceSynchronize());

    compute_buf = buf_u; 
    exchange_buf = buf_v;

    rreq_idx = 0;
    sreq_idx = 0;
    complete_sreq_idx = 0;
    complete_rreq_idx = 0;
    rreq_inflight = 0;
    sreq_inflight = 0;
    complete_sync_inflight = 0;
    complete_sync_idx = 0;
    complete_sync_wait = 0;

    int doublebuf_idx;

    /*warmup steps*/
    prepost_depth = (prepost_depth_global < iter_warmup) ? prepost_depth_global : iter_warmup;
    for (i=0; i<prepost_depth; i++) {
        doublebuf_idx = i%buffering;
        packbuf_disp = 0;
        /*y dim*/
        if(use_mpi == 0)
        {
            if (left != MPI_PROC_NULL) {
                mp_irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                         msg_size_bytes, left, &reg_unpack[doublebuf_idx],
                         &rreq[rreq_idx]);
                rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight++;
            }
            packbuf_disp += msg_size;
            if (right != MPI_PROC_NULL) {
                mp_irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                        msg_size_bytes, right, &reg_unpack[doublebuf_idx],
                        &rreq[rreq_idx]);
                rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight++;
            }
            packbuf_disp += msg_size;
            /*x dim*/
            if (bottom != MPI_PROC_NULL) {
                mp_irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                        msg_size_bytes, bottom, &reg_unpack[doublebuf_idx],
                        &rreq[rreq_idx]);
                rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight++;
            }
            packbuf_disp += msg_size;
            if (top != MPI_PROC_NULL) {
                mp_irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                        msg_size_bytes, top, &reg_unpack[doublebuf_idx],
                        &rreq[rreq_idx]);
                rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight++;
            }
        }
        else
        {
            if (left != MPI_PROC_NULL) {
                //printf("comm_rank: %d, left: %d\n", comm_rank, left);
                MPI_Irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                    msg_size, MPI_FLOAT, left, left, comm2d,
                    &mpi_rreq[rreq_idx]);
                rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight++;
            }
            packbuf_disp += msg_size;
            if (right != MPI_PROC_NULL) {
                //printf("comm_rank: %d, right: %d\n", comm_rank, right);
                MPI_Irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                    msg_size, MPI_FLOAT, right, right, comm2d,
                    &mpi_rreq[rreq_idx]);
                rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight++;
            }
            packbuf_disp += msg_size;
            /*x dim*/
            if (bottom != MPI_PROC_NULL) {
                //printf("comm_rank: %d, bottom: %d\n", comm_rank, bottom);
                MPI_Irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                    msg_size, MPI_FLOAT, bottom, bottom, comm2d,
                    &mpi_rreq[rreq_idx]);
                rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight++;
            }
            packbuf_disp += msg_size;
            if (top != MPI_PROC_NULL) {
                //printf("comm_rank: %d, top: %d\n", comm_rank, top);
                MPI_Irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                    msg_size, MPI_FLOAT, top, top, comm2d,
                    &mpi_rreq[rreq_idx]);
                rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight++;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < iter_warmup; i++) {

        //if (!i) //printf("warming up\n");
        doublebuf_idx = i%buffering;

        /*pack data*/
       	boundary_pack (packbuf[doublebuf_idx], exchange_buf, size, threadsperblock, boundary_sendrecv_stream);

        /*launch interior compute*/
        if (!use_single_stream)
            interior_compute (compute_buf, exchange_buf, size, boundary, threadsperblock, gridsize, interior_stream);

        /*post all sends*/ 
        packbuf_disp = 0;
	
    	if (use_async) {
                /*prepare and post requires requests to be contiguous, check if there is a wrap around*/
                int sreq_idx_start = sreq_idx;
                if ((sreq_idx_start + req_per_step) > sreq_max_inflight) {
                    fprintf(stderr, "sreq status array looping around in middle of a step: sreq_idx_start: %d sreq_max_inflight: %d \n",
                                sreq_idx_start, sreq_max_inflight);
                    exit(-1);
                }
     
    	       /*prepare all sends*/
                /*y dim*/
                if (right != MPI_PROC_NULL) {
                    mp_send_prepare((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                            msg_size_bytes, right, &reg_pack[doublebuf_idx],
                            &sreq[sreq_idx]);
                    sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight++;
                }
                packbuf_disp += msg_size;
                if (left != MPI_PROC_NULL) {
                    mp_send_prepare((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                            msg_size_bytes, left, &reg_pack[doublebuf_idx], 
                	    &sreq[sreq_idx]);
                    sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight++;
                }
                packbuf_disp += msg_size;
                /*x dim*/
                if (top != MPI_PROC_NULL) {
                    mp_send_prepare((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                            msg_size_bytes, top, &reg_pack[doublebuf_idx],
                            &sreq[sreq_idx]);
                    sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight++;
                }
                packbuf_disp += msg_size;
                if (bottom != MPI_PROC_NULL) {
                    mp_send_prepare((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                            msg_size_bytes, bottom, &reg_pack[doublebuf_idx],
                            &sreq[sreq_idx]);
                    sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight++;
                }

    	       /*post sends on stream*/
                mp_isend_post_all_on_stream (req_per_step, &sreq[sreq_idx_start], boundary_sendrecv_stream);

                /*launch interior compute*/
                if (use_single_stream)
                     interior_compute (compute_buf, exchange_buf, size, boundary, threadsperblock, gridsize, boundary_sendrecv_stream);

                /*post wait for recv on the stream*/
                wait_idx = (i*req_per_step)%rreq_max_inflight;
                mp_wait_all_on_stream(req_per_step, &rreq[wait_idx], boundary_sendrecv_stream);
    	} else {
            CUDA_CHECK(cudaStreamSynchronize(boundary_sendrecv_stream));

            /*launch interior compute*/
            if (use_single_stream)
                interior_compute (compute_buf, exchange_buf, size, boundary, threadsperblock, gridsize, boundary_sendrecv_stream);

            /*post sends*/
            if(use_mpi == 0)
            {
                /*y dim*/
                if (right != MPI_PROC_NULL) {
                    mp_isend((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                            msg_size_bytes, right, &reg_pack[doublebuf_idx],
                            &sreq[sreq_idx]);
                    sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight++;
                }
                packbuf_disp += msg_size;
                if (left != MPI_PROC_NULL) {
                    mp_isend((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                            msg_size_bytes, left, &reg_pack[doublebuf_idx],
                        &sreq[sreq_idx]);
                    sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight++;
                }
                packbuf_disp += msg_size;
                /*x dim*/
                if (top != MPI_PROC_NULL) {
                    mp_isend((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                            msg_size_bytes, top, &reg_pack[doublebuf_idx],
                            &sreq[sreq_idx]);
                    sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight++;
                }
                packbuf_disp += msg_size;
                if (bottom != MPI_PROC_NULL) {
                    mp_isend((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                            msg_size_bytes, bottom, &reg_pack[doublebuf_idx],
                            &sreq[sreq_idx]);
                    sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight++;
                }

                complete_idx = 0;
                while (complete_idx < req_per_step) {
                    complete_rreq_idx = (i%prepost_depth)*req_per_step + complete_idx;
                    mp_wait(&rreq[complete_rreq_idx]);
                    complete_idx++;
                    rreq_inflight--;
                }
     
                complete_idx = 0;
                while (complete_idx < sreq_inflight) {
                    mp_wait(&sreq[complete_idx]);
                    complete_idx++;
                }

                sreq_inflight = 0;
            }
            else
            {
                /*y dim*/
                if (right != MPI_PROC_NULL) {
                    MPI_Isend((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                            msg_size, MPI_FLOAT, right, comm_rank, comm2d,
                            &mpi_sreq[sreq_idx]);
                    sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight++;
                }
                packbuf_disp += msg_size;
                if (left != MPI_PROC_NULL) {
                    MPI_Isend((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                            msg_size, MPI_FLOAT, left, comm_rank, comm2d,
                            &mpi_sreq[sreq_idx]);
                    sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight++;
                }
                packbuf_disp += msg_size;
                /*x dim*/
                if (top != MPI_PROC_NULL) {
                    MPI_Isend((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                            msg_size, MPI_FLOAT, top, comm_rank, comm2d,
                            &mpi_sreq[sreq_idx]);
                    sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight++;
                }
                packbuf_disp += msg_size;
                if (bottom != MPI_PROC_NULL) {
                    MPI_Isend((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                        msg_size, MPI_FLOAT, bottom, comm_rank, comm2d,
                        &mpi_sreq[sreq_idx]);
                    sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight++;
                }

                complete_idx = 0;
                while (complete_idx < req_per_step) {
                    complete_rreq_idx = (i%prepost_depth)*req_per_step + complete_idx;
                    MPI_Wait(&mpi_rreq[complete_rreq_idx], &mpi_rstatus[complete_rreq_idx]);
                    complete_idx++;
                    rreq_inflight--;
                }
     
                complete_idx = 0;
                while (complete_idx < sreq_inflight) {
                    MPI_Wait(&mpi_sreq[complete_idx], &mpi_sstatus[complete_idx]);
                    complete_idx++;
                }

                sreq_inflight = 0;

               // MPI_Waitall(rreq_idx, rreq, MPI_STATUS_IGNORE);
                //MPI_Waitall(sreq_idx, sreq, MPI_STATUS_IGNORE);
            }
        }

        /*unpack data*/
        boundary_unpack (exchange_buf, unpackbuf[doublebuf_idx], size, threadsperblock, boundary_sendrecv_stream);

        /*launch boundary computation*/
        boundary_compute (compute_buf, exchange_buf, size, boundary, threadsperblock, gridsize, boundary_sendrecv_stream);

	/*synchronize at end of each step*/
	if (use_async) {
            CUDA_CHECK(cudaEventRecord(sendrecv_sync_event[complete_sync_idx], boundary_sendrecv_stream));
            if (!use_single_stream) {
                CUDA_CHECK(cudaEventRecord(interior_sync_event[complete_sync_idx], interior_stream));

                CUDA_CHECK(cudaStreamWaitEvent(boundary_sendrecv_stream, interior_sync_event[complete_sync_idx], 0));
                CUDA_CHECK(cudaStreamWaitEvent(interior_stream, sendrecv_sync_event[complete_sync_idx], 0));
	    }

	    complete_sync_inflight++;
            complete_sync_idx = (complete_sync_idx + 1)%(batches_inflight*steps_per_batch);

            assert(complete_sync_inflight <= batches_inflight*steps_per_batch);

	    /*synchronize on oldest batch*/
            if (complete_sync_inflight == batches_inflight*steps_per_batch) {
	        if (tracking_events) {
                    complete_sync_wait = (complete_sync_wait + steps_per_batch - 1)%(batches_inflight*steps_per_batch);

                    CUDA_CHECK(cudaEventSynchronize(sendrecv_sync_event[complete_sync_wait]));
                    if (!use_single_stream) { 
                        CUDA_CHECK(cudaEventSynchronize(interior_sync_event[complete_sync_wait]));	
		    }
                    complete_sync_wait = (complete_sync_wait + 1)%(batches_inflight*steps_per_batch);
		}
	        complete_sync_inflight = complete_sync_inflight - steps_per_batch; 

                while (rreq_inflight > req_per_step*(steps_per_batch*(batches_inflight - 1) + prepost_depth)) {
                    mp_wait(&rreq[complete_rreq_idx]);
                    complete_rreq_idx = (complete_rreq_idx + 1)%rreq_max_inflight;
                    rreq_inflight--; 
                }

                while (sreq_inflight > req_per_step*steps_per_batch*(batches_inflight - 1)) {
                    mp_wait(&sreq[complete_sreq_idx]);
                    complete_sreq_idx = (complete_sreq_idx + 1)%sreq_max_inflight; 
                    sreq_inflight--;
                }
	    }
	} else {
	    if (!force_stream_sync) { 
                CUDA_CHECK(cudaDeviceSynchronize());
	    } else { 
                CUDA_CHECK(cudaEventRecord(sendrecv_sync_event[0], boundary_sendrecv_stream));
                if (!use_single_stream) {
                    CUDA_CHECK(cudaEventRecord(interior_sync_event[0], interior_stream));

                    CUDA_CHECK(cudaStreamWaitEvent(boundary_sendrecv_stream, interior_sync_event[0], 0));
                    CUDA_CHECK(cudaStreamWaitEvent(interior_stream, sendrecv_sync_event[0], 0));
	        }

                CUDA_CHECK(cudaEventSynchronize(sendrecv_sync_event[0]));
                if (!use_single_stream) {
                    CUDA_CHECK(cudaEventSynchronize(interior_sync_event[0]));
                }
	    }
	}

        /*prepost recv for a future step*/ 
        if ((i+prepost_depth) < iter_warmup) {
            doublebuf_idx = (i+prepost_depth)%buffering;
            packbuf_disp = 0;
            if(use_mpi == 0)
            {
                /*y dim*/
                if (left != MPI_PROC_NULL) {
                 mp_irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp), 
                	     msg_size_bytes, left, &reg_unpack[doublebuf_idx], 
                	     &rreq[rreq_idx]);
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                 rreq_inflight++;
                }
                packbuf_disp += msg_size;
                if (right != MPI_PROC_NULL) { 
                 mp_irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                         msg_size_bytes, right, &reg_unpack[doublebuf_idx], 
                	    &rreq[rreq_idx]);
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                 rreq_inflight++;
                }
                packbuf_disp += msg_size;
                /*x dim*/
                if (bottom != MPI_PROC_NULL) {
                 mp_irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                         msg_size_bytes, bottom, &reg_unpack[doublebuf_idx],
                         &rreq[rreq_idx]);
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                 rreq_inflight++;
                }
                packbuf_disp += msg_size;
                if (top != MPI_PROC_NULL) {
                 mp_irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                         msg_size_bytes, top, &reg_unpack[doublebuf_idx],
                         &rreq[rreq_idx]);
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                 rreq_inflight++;
                }
            }
            else
            {
                if (left != MPI_PROC_NULL) {
                    //printf("1 comm_rank: %d, left: %d\n", comm_rank, left);
                    MPI_Irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                        msg_size, MPI_FLOAT, left, left, comm2d,
                        &mpi_rreq[rreq_idx]);
                    rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                    rreq_inflight++;
                }
                packbuf_disp += msg_size;
                if (right != MPI_PROC_NULL) {
                    //printf("1 comm_rank: %d, right: %d\n", comm_rank, right);
                    MPI_Irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                        msg_size, MPI_FLOAT, right, right, comm2d,
                        &mpi_rreq[rreq_idx]);
                    rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                    rreq_inflight++;
                }
                packbuf_disp += msg_size;
                /*x dim*/
                if (bottom != MPI_PROC_NULL) {
                    //printf("1 comm_rank: %d, bottom: %d\n", comm_rank, bottom);
                    MPI_Irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                        msg_size, MPI_FLOAT, bottom, bottom, comm2d,
                        &mpi_rreq[rreq_idx]);
                    rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                    rreq_inflight++;
                }
                packbuf_disp += msg_size;
                if (top != MPI_PROC_NULL) {
                    //printf("1 comm_rank: %d, top: %d\n", comm_rank, top);
                    MPI_Irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                        msg_size, MPI_FLOAT, top, top, comm2d,
                        &mpi_rreq[rreq_idx]);
                    rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                    rreq_inflight++;
                }
            }
	}

	/*if its the last iteration, synchronize on all batches*/
        if (use_async) {
            if (i == (iter_warmup - 1)) {
                if (complete_sync_inflight > 0) {
                    if (tracking_events) {
                        complete_sync_wait = (complete_sync_wait + complete_sync_inflight - 1)%(batches_inflight*steps_per_batch);
                        CUDA_CHECK(cudaEventSynchronize(sendrecv_sync_event[complete_sync_wait]));
                        if (!use_single_stream)
                            CUDA_CHECK(cudaEventSynchronize(interior_sync_event[complete_sync_wait]));
                    }		
                    complete_sync_inflight = 0;
                }
  
                while (rreq_inflight > 0) {
                    mp_wait(&rreq[complete_rreq_idx]);
                    complete_rreq_idx = (complete_rreq_idx + 1)%rreq_max_inflight;
                    rreq_inflight--;
                }
  
                while (sreq_inflight > 0) { 
                    mp_wait(&sreq[complete_sreq_idx]);
                    complete_sreq_idx = (complete_sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight--;
                }
            }
        }

    	/*interchange the compute and communication buffers*/
        temp = exchange_buf; 
    	exchange_buf = compute_buf; 
    	compute_buf = temp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (wait_for_key && !comm_rank) {
        int c;
        puts("press any key");
        c = getchar();
    }

#ifdef _REPEAT_TIMED_LOOP_
    int rep;
    for (rep = 0; rep < reps_count; rep++) { 
#endif

    rreq_idx = 0;
    sreq_idx = 0;
    complete_sreq_idx = 0;
    complete_rreq_idx = 0;
    rreq_inflight = 0;
    sreq_inflight = 0;
    complete_sync_inflight = 0;
    complete_sync_idx = 0;
    complete_sync_wait = 0;

    /*timed iterations*/
    time_start = cycles_to_ns(get_cycles());
#ifndef _FREE_NETWORK_
    prepost_depth = (prepost_depth_global < iter_count) ? prepost_depth_global : iter_count;
    for (i=0; i<prepost_depth; i++) {
        doublebuf_idx = i%buffering;
        packbuf_disp = 0;
        if(use_mpi == 0)
        {
            /*y dim*/
            if (left != MPI_PROC_NULL) {
                mp_irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                         msg_size_bytes, left, &reg_unpack[doublebuf_idx],
                         &rreq[rreq_idx]);
                rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight++;
            }
            packbuf_disp += msg_size;
            if (right != MPI_PROC_NULL) {
                mp_irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                        msg_size_bytes, right, &reg_unpack[doublebuf_idx],
                        &rreq[rreq_idx]);
                rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight++;
            }
            packbuf_disp += msg_size;
            /*x dim*/
            if (bottom != MPI_PROC_NULL) {
                mp_irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                        msg_size_bytes, bottom, &reg_unpack[doublebuf_idx],
                        &rreq[rreq_idx]);
                rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight++;
            }
            packbuf_disp += msg_size;
            if (top != MPI_PROC_NULL) {
                mp_irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                        msg_size_bytes, top, &reg_unpack[doublebuf_idx],
                        &rreq[rreq_idx]);
                rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight++;
            }
        }
        else
        {
            if (left != MPI_PROC_NULL) {
                //printf("2 comm_rank: %d, left: %d\n", comm_rank, left);
                MPI_Irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                    msg_size, MPI_FLOAT, left, left, comm2d,
                    &mpi_rreq[rreq_idx]);
                rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight++;
            }
            packbuf_disp += msg_size;
            if (right != MPI_PROC_NULL) {
                //printf("2 comm_rank: %d, right: %d\n", comm_rank, right);
                MPI_Irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                    msg_size, MPI_FLOAT, right, right, comm2d,
                    &mpi_rreq[rreq_idx]);
                rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight++;
            }
            packbuf_disp += msg_size;
            /*x dim*/
            if (bottom != MPI_PROC_NULL) {
                //printf("2 comm_rank: %d, bottom: %d\n", comm_rank, bottom);
                MPI_Irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                    msg_size, MPI_FLOAT, bottom, bottom, comm2d,
                    &mpi_rreq[rreq_idx]);
                rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight++;
            }
            packbuf_disp += msg_size;
            if (top != MPI_PROC_NULL) {
                //printf("2 comm_rank: %d, top: %d\n", comm_rank, top);
                MPI_Irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                    msg_size, MPI_FLOAT, top, top, comm2d,
                    &mpi_rreq[rreq_idx]);
                rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight++;
            }
        }
    }
#endif
    time_stop = cycles_to_ns(get_cycles());
    time_prepost = (time_stop - time_start); 

#if defined (_ENABLE_DRPROF_)
    char *tags;
    if (use_async) { 	
        tags = "pack|interior|prepsend|postsend|postwait|unpack|boundary|postsync|waitsync|postrecv|";
    } else { 
        tags = "pack|interior|packsync|postsend|wait|unpack|boundary|gpusync|postrecv";
    }

    if (prof_init(&prof, 1000,  1000, "1us", 50, 1, tags)) {
        fprintf(stderr, "error in prof_init init.\n");
        exit(-1);
    }
    prof_start = 1;
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    CUDA_CHECK(cudaEventRecord(start_event, 0));

    for (i = 0; i < iter_count; i++) {
	doublebuf_idx = i%buffering;

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);        
#endif

#ifndef _FREE_PACK_
        /*pack data*/
       	boundary_pack (packbuf[doublebuf_idx], exchange_buf, size, threadsperblock, boundary_sendrecv_stream);
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif

#ifndef _FREE_INTERIOR_COMPUTE_
        /*launch interior compute*/
        if (!use_single_stream)
            interior_compute (compute_buf, exchange_buf, size, boundary, threadsperblock, gridsize, interior_stream);
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif

        /*post all sends*/ 
        packbuf_disp = 0;

#ifndef _FREE_NETWORK_
	if (use_async) { 
	    /*prepare and post requires requests to be contiguous, check if there is a wrap around*/
	    int sreq_idx_start = sreq_idx;
	    if ((sreq_idx_start + req_per_step) > sreq_max_inflight) {
	        fprintf(stderr, "sreq status array looping around in middle of a step: sreq_idx_start: %d sreq_max_inflight: %d \n", 
	    		sreq_idx_start, sreq_max_inflight);
	        exit(-1);
	    }

	    /*prepare all sends*/
            /*y dim*/
            if (right != MPI_PROC_NULL) {
                mp_send_prepare((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                        msg_size_bytes, right, &reg_pack[doublebuf_idx],
                        &sreq[sreq_idx]);
                sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                sreq_inflight++;
            }
            packbuf_disp += msg_size;
            if (left != MPI_PROC_NULL) {
                mp_send_prepare((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                        msg_size_bytes, left, &reg_pack[doublebuf_idx], 
            	    &sreq[sreq_idx]);
                sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                sreq_inflight++;
            }
            packbuf_disp += msg_size;
            /*x dim*/
            if (top != MPI_PROC_NULL) {
                mp_send_prepare((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                        msg_size_bytes, top, &reg_pack[doublebuf_idx],
                        &sreq[sreq_idx]);
                sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                sreq_inflight++;
            }
            packbuf_disp += msg_size;
            if (bottom != MPI_PROC_NULL) {
                mp_send_prepare((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                        msg_size_bytes, bottom, &reg_pack[doublebuf_idx],
                        &sreq[sreq_idx]);
                sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                sreq_inflight++;
            }

#if defined (_ENABLE_DRPROF_)
            PROF(&prof, prof_idx++);
#endif

            /*post sends on stream*/
            mp_isend_post_all_on_stream (req_per_step, &sreq[sreq_idx_start], boundary_sendrecv_stream);

#if defined (_ENABLE_DRPROF_)
            PROF(&prof, prof_idx++);
#endif

            /*launch interior compute*/
            if (use_single_stream)
                interior_compute (compute_buf, exchange_buf, size, boundary, threadsperblock, gridsize, boundary_sendrecv_stream);

            /*post wait for recv on the stream*/
            wait_idx = (i*req_per_step)%rreq_max_inflight;
            mp_wait_all_on_stream(req_per_step, &rreq[wait_idx], boundary_sendrecv_stream);
	} else {
            CUDA_CHECK(cudaStreamSynchronize(boundary_sendrecv_stream));

            /*launch interior compute*/
            if (use_single_stream)
                interior_compute (compute_buf, exchange_buf, size, boundary, threadsperblock, gridsize, boundary_sendrecv_stream);

            if(use_mpi == 0)
            {
                /*post all sends*/
                /*y dim*/
                if (right != MPI_PROC_NULL) {
                    mp_isend((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                            msg_size_bytes, right, &reg_pack[doublebuf_idx],
                            &sreq[sreq_idx]);
                    sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight++;
                }
                packbuf_disp += msg_size;
                if (left != MPI_PROC_NULL) {
                    mp_isend((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                            msg_size_bytes, left, &reg_pack[doublebuf_idx],
                        &sreq[sreq_idx]);
                    sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight++;
                }
                packbuf_disp += msg_size;
                /*x dim*/
                if (top != MPI_PROC_NULL) {
                    mp_isend((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                            msg_size_bytes, top, &reg_pack[doublebuf_idx],
                            &sreq[sreq_idx]);
                    sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight++;
                }
                packbuf_disp += msg_size;
                if (bottom != MPI_PROC_NULL) {
                    mp_isend((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                            msg_size_bytes, bottom, &reg_pack[doublebuf_idx],
                            &sreq[sreq_idx]);
                    sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight++;
                }


#if defined (_ENABLE_DRPROF_)
            PROF(&prof, prof_idx++);
#endif

            complete_idx = 0;
            while (complete_idx < req_per_step) {
                complete_rreq_idx = (i%prepost_depth)*req_per_step + complete_idx;
                mp_wait(&rreq[complete_rreq_idx]);
                complete_idx++;
                rreq_inflight--;
            }

            complete_idx = 0;
            while (complete_idx < sreq_inflight) {
                mp_wait(&sreq[complete_idx]);
                complete_idx++;
            }

            sreq_inflight = 0;
        }
        else
        {
            /*y dim*/
            if (right != MPI_PROC_NULL) {
                MPI_Isend((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                        msg_size, MPI_FLOAT, right, comm_rank, comm2d,
                        &mpi_sreq[sreq_idx]);
                sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                sreq_inflight++;
            }
            packbuf_disp += msg_size;
            if (left != MPI_PROC_NULL) {
                MPI_Isend((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                        msg_size, MPI_FLOAT, left, comm_rank, comm2d,
                        &mpi_sreq[sreq_idx]);
                sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                sreq_inflight++;
            }
            packbuf_disp += msg_size;
            /*x dim*/
            if (top != MPI_PROC_NULL) {
                MPI_Isend((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                        msg_size, MPI_FLOAT, top, comm_rank, comm2d,
                        &mpi_sreq[sreq_idx]);
                sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                sreq_inflight++;
            }
            packbuf_disp += msg_size;
            if (bottom != MPI_PROC_NULL) {
                MPI_Isend((void *)(packbuf[doublebuf_idx] + packbuf_disp),
                    msg_size, MPI_FLOAT, bottom, comm_rank, comm2d,
                    &mpi_sreq[sreq_idx]);
                sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
                sreq_inflight++;
            }

            complete_idx = 0;
            while (complete_idx < req_per_step) {
                complete_rreq_idx = (i%prepost_depth)*req_per_step + complete_idx;
                MPI_Wait(&mpi_rreq[complete_rreq_idx], &mpi_rstatus[complete_rreq_idx]);
                complete_idx++;
                rreq_inflight--;
            }
 
            complete_idx = 0;
            while (complete_idx < sreq_inflight) {
                MPI_Wait(&mpi_sreq[complete_idx], &mpi_sstatus[complete_idx]);
                complete_idx++;
            }

            sreq_inflight = 0;

           // MPI_Waitall(rreq_idx, rreq, MPI_STATUS_IGNORE);
            //MPI_Waitall(sreq_idx, sreq, MPI_STATUS_IGNORE);
        }
	}
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif

#ifndef _FREE_PACK_
        /*unpack data*/
        boundary_unpack (exchange_buf, unpackbuf[doublebuf_idx], size, threadsperblock, boundary_sendrecv_stream);
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif

#ifndef _FREE_BOUNDARY_COMPUTE_
	/*launch boundary computation*/
        boundary_compute (compute_buf, exchange_buf, size, boundary, threadsperblock, gridsize, boundary_sendrecv_stream);
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif

	if (use_async) {
#ifndef _FREE_SYNC_
            CUDA_CHECK(cudaEventRecord(sendrecv_sync_event[complete_sync_idx], boundary_sendrecv_stream));
            if (!use_single_stream) {
                CUDA_CHECK(cudaEventRecord(interior_sync_event[complete_sync_idx], interior_stream));

                CUDA_CHECK(cudaStreamWaitEvent(boundary_sendrecv_stream, interior_sync_event[complete_sync_idx], 0));
                CUDA_CHECK(cudaStreamWaitEvent(interior_stream, sendrecv_sync_event[complete_sync_idx], 0));
	    }
#endif
	    complete_sync_inflight++;
            complete_sync_idx = (complete_sync_idx + 1)%(batches_inflight*steps_per_batch);

#if defined (_ENABLE_DRPROF_)
            PROF(&prof, prof_idx++);
#endif

            assert(complete_sync_inflight <= batches_inflight*steps_per_batch);
	    /*synchronize on oldest batch*/
            if (complete_sync_inflight == batches_inflight*steps_per_batch) {

                if (tracking_events) {
                    complete_sync_wait = (complete_sync_wait + steps_per_batch - 1)%(batches_inflight*steps_per_batch);

#ifndef _FREE_SYNC_
                    CUDA_CHECK(cudaEventSynchronize(sendrecv_sync_event[complete_sync_wait]));
                    if (!use_single_stream)
                        CUDA_CHECK(cudaEventSynchronize(interior_sync_event[complete_sync_wait]));
#endif
                    complete_sync_wait = (complete_sync_wait + 1)%(batches_inflight*steps_per_batch);
		}
	        complete_sync_inflight = complete_sync_inflight - steps_per_batch; 

#ifndef _FREE_NETWORK_
            while (rreq_inflight > req_per_step*(steps_per_batch*(batches_inflight - 1) + prepost_depth)) {
                mp_wait(&rreq[complete_rreq_idx]);
                complete_rreq_idx = (complete_rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight--; 
            }

            while (sreq_inflight > req_per_step*steps_per_batch*(batches_inflight - 1)) {
                mp_wait(&sreq[complete_sreq_idx]);
                complete_sreq_idx = (complete_sreq_idx + 1)%sreq_max_inflight; 
                sreq_inflight--;
            }
#endif
	    }
	} else {
#ifndef _FREE_SYNC_
            CUDA_CHECK(cudaDeviceSynchronize());
#endif
	}

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif

        /*prepost recv for a future step*/ 
#ifndef _FREE_NETWORK_
        if ((i+prepost_depth) < iter_count) {
            doublebuf_idx = (i+prepost_depth)%buffering;
            packbuf_disp = 0;
            if(use_mpi == 0)
            {
                /*y dim*/
                if (left != MPI_PROC_NULL) {
                 mp_irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp), 
                	     msg_size_bytes, left, &reg_unpack[doublebuf_idx], 
                	     &rreq[rreq_idx]);
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                 rreq_inflight++;
                }
                packbuf_disp += msg_size;
                if (right != MPI_PROC_NULL) { 
                 mp_irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                         msg_size_bytes, right, &reg_unpack[doublebuf_idx], 
                	    &rreq[rreq_idx]);
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                 rreq_inflight++;
                }
                packbuf_disp += msg_size;
                /*x dim*/
                if (bottom != MPI_PROC_NULL) {
                 mp_irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                         msg_size_bytes, bottom, &reg_unpack[doublebuf_idx],
                         &rreq[rreq_idx]);
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                 rreq_inflight++;
                }
                packbuf_disp += msg_size;
                if (top != MPI_PROC_NULL) {
                 mp_irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                         msg_size_bytes, top, &reg_unpack[doublebuf_idx],
                         &rreq[rreq_idx]);
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                 rreq_inflight++;
                } 
            }
            else
            {
                if (left != MPI_PROC_NULL) {
                    //printf("3 comm_rank: %d, left: %d\n", comm_rank, left);
                    MPI_Irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                        msg_size, MPI_FLOAT, left, left, comm2d,
                        &mpi_rreq[rreq_idx]);
                    rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                    rreq_inflight++;
                }
                packbuf_disp += msg_size;
                if (right != MPI_PROC_NULL) {
                    //printf("3 comm_rank: %d, right: %d\n", comm_rank, right);
                    MPI_Irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                        msg_size, MPI_FLOAT, right, right, comm2d,
                        &mpi_rreq[rreq_idx]);
                    rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                    rreq_inflight++;
                }
                packbuf_disp += msg_size;
                /*x dim*/
                if (bottom != MPI_PROC_NULL) {
                    //printf("3 comm_rank: %d, bottom: %d\n", comm_rank, bottom);
                    MPI_Irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                        msg_size, MPI_FLOAT, bottom, bottom, comm2d,
                        &mpi_rreq[rreq_idx]);
                    rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                    rreq_inflight++;
                }
                packbuf_disp += msg_size;
                if (top != MPI_PROC_NULL) {
                    //printf("3 comm_rank: %d, top: %d\n", comm_rank, top);
                    MPI_Irecv((void *)(unpackbuf[doublebuf_idx] + packbuf_disp),
                        msg_size, MPI_FLOAT, top, top, comm2d,
                        &mpi_rreq[rreq_idx]);
                    rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                    rreq_inflight++;
                }
            }
	}
#endif 

	if (use_async) { 
	    /*if its the last iteration, synchronize on all batches*/
            if (i == (iter_count - 1)) {
                if (complete_sync_inflight > 0) {

                    if (tracking_events) {
                        complete_sync_wait = (complete_sync_wait + complete_sync_inflight - 1)%(batches_inflight*steps_per_batch);
#ifndef _FREE_SYNC_
                        CUDA_CHECK(cudaEventSynchronize(sendrecv_sync_event[complete_sync_wait]));
		        if (!use_single_stream)
                             CUDA_CHECK(cudaEventSynchronize(interior_sync_event[complete_sync_wait]));
#endif
		    }

                    complete_sync_inflight = 0;
                }

 #ifndef _FREE_NETWORK_
                while (rreq_inflight > 0) {
                    mp_wait(&rreq[complete_rreq_idx]);
                    complete_rreq_idx = (complete_rreq_idx + 1)%rreq_max_inflight;
                    rreq_inflight--;
                }
  
                while (sreq_inflight > 0) { 
                    mp_wait(&sreq[complete_sreq_idx]);
                    complete_sreq_idx = (complete_sreq_idx + 1)%sreq_max_inflight;
                    sreq_inflight--;
                }
#endif
	    }
	}

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
        prof_update(&prof);
        prof_idx = 0;
#endif

	/*interchange the compute and communication buffers*/
        temp = exchange_buf; 
	exchange_buf = compute_buf; 
	compute_buf = temp;
    }

    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&time_elapsed, start_event, stop_event));
    if (comm_rank == 0) {
#ifdef _REPEAT_TIMED_LOOP_
        fprintf(stdout, "%3lld %3lld %8.2lf usec \n", size, rep, (time_elapsed*1e3 + (time_prepost/1000))/iter_count);
#else
        //fprintf(stdout, "%3lld %8.2lf usec \n", size, (time_elapsed*1e3 + (time_prepost/1000))/iter_count);
        fprintf(stdout, "%8.2lf usec \n", (time_elapsed*1e3 + (time_prepost/1000))/iter_count);
#endif

#if defined (_ENABLE_DRPROF_)
	fprintf(stdout, "Prepost depth: %d latency: %ld nsec \n", prepost_depth, time_prepost);
        prof_dump(&prof);
#if defined (_ENABLE_DRPROF_ISEND_)
        prof_dump(&prof_isend_prepare);
#endif
#endif
    }

    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef _REPEAT_TIMED_LOOP_
    }
#endif

#ifdef _VALIDATE_
#ifdef _REPEAT_TIMED_LOOP_
    iter_count = iter_count*reps_count;
#endif
    emulate_on_host (buf_u_h, buf_v_h, size, boundary, 1/*ghost*/, 
            comm_rank, left, right, bottom, top, (iter_count + iter_warmup), comm2d);

    if(use_gpu_comm_buffers == 0)
    {
        memcpy(buf_u_h, buf_u, buf_size);
        memcpy(buf_v_h, buf_v, buf_size);
    }
    else
    {
        CUDA_CHECK(cudaMemcpy(buf_u_h, buf_u, buf_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(buf_v_h, buf_v, buf_size, cudaMemcpyDeviceToHost));
    }

    validate (buf_u_h, buf_v_h, size, 1);
#endif

    if(use_gpu_comm_buffers == 0)
    {
        CUDA_CHECK(cudaFreeHost(buf_v));
        CUDA_CHECK(cudaFreeHost(buf_u));
        for (i=0; i<buffering; i++) { 
            CUDA_CHECK(cudaFreeHost(packbuf[i]));
            if(use_mpi == 0)
                mp_deregister(&reg_pack[i]);

            CUDA_CHECK(cudaFreeHost(unpackbuf[i]));
            if(use_mpi == 0)
                mp_deregister(&reg_unpack[i]);
        }
    }
    else
    {
        CUDA_CHECK(cudaFree(buf_u));
        CUDA_CHECK(cudaFree(buf_v));
        for (i=0; i<buffering; i++) { 
            CUDA_CHECK(cudaFree(packbuf[i]));
            if(use_mpi == 0)
                mp_deregister(&reg_pack[i]);

            CUDA_CHECK(cudaFree(unpackbuf[i]));
            if(use_mpi == 0)
                mp_deregister(&reg_unpack[i]);
        }
    }

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaStreamDestroy(interior_stream));
    CUDA_CHECK(cudaStreamDestroy(boundary_sendrecv_stream));
    for (i=0; i<batches_inflight*steps_per_batch; i++) {
        CUDA_CHECK(cudaEventDestroy(sendrecv_sync_event[i]));
        CUDA_CHECK(cudaEventDestroy(interior_sync_event[i]));
    }

    if(use_mpi == 0)
    {
        free(sreq);
        free(rreq);
    }

    return 0;
}

int main (int c, char *v[])
{
    int iter_count, iter_count_large, iter_warmup, iter_warmup_large;
    int dim[2], period[2];
    int reorder, boundary_compute_width;
    MPI_Comm comm2d;

    px=4;
    py=4;
#if defined (_VALIDATE_)
    iter_count=iter_count_large=30;
    iter_warmup=iter_warmup_large=10;
#else
    iter_count=iter_count_large=200;
    iter_warmup=iter_warmup_large=20;
#endif
    boundary_compute_width = default_boundary;

    if (getenv("WAIT_FOR_KEY") != NULL) {
        wait_for_key = atoi(getenv("WAIT_FOR_KEY"));
    }
    if (getenv("PX") != NULL) {
        px = atoi(getenv("PX"));
    }
    if (getenv("PY") != NULL) {
        py = atoi(getenv("PY"));
    }
    if (getenv("MAX_SIZE") != NULL) {
        max_size = atoi(getenv("MAX_SIZE"));
    }
    if (getenv("MIN_SIZE") != NULL) {
        min_size = atoi(getenv("MIN_SIZE"));
    }
    if (getenv("ITER_COUNT") != NULL) {
        iter_count = atoi(getenv("ITER_COUNT"));
    }
    if (getenv("WARMUP_COUNT") != NULL) {
        iter_warmup = atoi(getenv("WARMUP_COUNT"));
    }
    if (getenv("ITER_COUNT_LARGE") != NULL) {
        iter_count_large = atoi(getenv("ITER_COUNT_LARGE"));
    }
    if (getenv("WARMUP_COUNT_LARGE") != NULL) {
        iter_warmup_large = atoi(getenv("WARMUP_COUNT_LARGE"));
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
    if (getenv("ITERS_PER_BATCH") != NULL) {
        steps_per_batch = atoi(getenv("ITERS_PER_BATCH"));
    }
    if (getenv("PREPOST_DEPTH") != NULL) {
        prepost_depth_global = atoi(getenv("PREPOST_DEPTH"));
    }
    if (getenv("BATCHES_INFLIGHT") != NULL) {
        batches_inflight = atoi(getenv("BATCHES_INFLIGHT"));
    }
    if (getenv("BUFFERING") != NULL) {
        buffering = atoi(getenv("BUFFERING"));
    }
    if (getenv("USE_PERIOD") != NULL) {
        period_value = atoi(getenv("USE_PERIOD"));
    }
    if (getenv("USE_GPU_ASYNC") != NULL) {
        use_async = atoi(getenv("USE_GPU_ASYNC"));
    }
    if (getenv("USE_SINGLE_STREAM") != NULL) {
        use_single_stream = atoi(getenv("USE_SINGLE_STREAM"));
    }
    if (getenv("FORCE_STREAM_SYNC") != NULL) {
        force_stream_sync = atoi(getenv("FORCE_STREAM_SYNC"));
    }

    if (getenv("REPS_COUNT") != NULL) {
        reps_count = atoi(getenv("REPS_COUNT"));
    }

    if (getenv("USE_GPU_COMM_BUFFERS")) {
      use_gpu_comm_buffers = atoi(getenv("USE_GPU_COMM_BUFFERS"));
      if(use_gpu_comm_buffers) printf("Warning: communcation buffers on GPU\n");
    }

    if (getenv("USE_MPI")) {
      use_mpi = atoi(getenv("USE_MPI"));
    }

    if(use_mpi ==1 && use_async == 1) {
        fprintf(stderr, "You cannot have both Async and MPI \n");
        exit(-1);
    }

    int dev_count = 0, dev_id = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0) {
        fprintf(stderr, "no CUDA devices found \n");
        exit(-1);
    }

    MPI_Init(&c, &v);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

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
    dev_id = local_rank%dev_count;
  
    if (getenv("USE_GPU") != NULL) { 
	dev_id = atoi(getenv("USE_GPU"));
	if (dev_id > dev_count) {
	    fprintf(stderr, "Invalid device ID, falling back to roundrobin binding\n");
	    dev_id = local_rank%dev_count;
	}
    }

    printf("[%d] async=%d single_stream=%d use_mpi=%d\n", comm_rank, use_async, use_single_stream, use_mpi);

    CUDA_CHECK(cudaSetDevice(dev_id));

    // print out info about my GPU
    struct cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev_id));
    printf("[%d] GPU%d %s\n", comm_rank, dev_id, deviceProp.name);

    /*create the stencil communicator*/
    dim[0]    = py;
    dim[1]    = px;
    period[0] = period_value;
    period[1] = period_value;
    reorder = 0;

    MPI_Cart_create (MPI_COMM_WORLD, 2, dim, period, reorder, &comm2d);
    MPI_Cart_shift(comm2d, 0,  1,  &left, &right );
    MPI_Cart_shift(comm2d, 1,  1,  &bottom, &top );
    MPI_Comm_rank(comm2d, &comm_rank);
    printf("[%d] left: %d right: %d top: %d bottom: %d iters_per_batch: %d batches_inflight: %d buffering: %d iter_count: %d iter_warmup: %d iter_count_large: %d iter_warmup_large: %d\n", comm_rank, left, right, top, bottom, steps_per_batch, batches_inflight, buffering, iter_count, iter_warmup, iter_count_large, iter_warmup_large);

    if (left != MPI_PROC_NULL) {
        req_per_step++; 
        peers[peer_count] = left;
        peer_count++;
    }
    if (right != MPI_PROC_NULL) {
        req_per_step++; 
        if (right != left) { 
            peers[peer_count] = right;
            peer_count++;
        }
    }
    if (top != MPI_PROC_NULL) { 
        req_per_step++; 
        peers[peer_count] = top;
        peer_count++;
    }
    if (bottom != MPI_PROC_NULL) { 
        req_per_step++; 
        if (bottom != top) { 
                peers[peer_count] = bottom;
                peer_count++;
        }
    }

    MPI_Cart_coords(comm2d, comm_rank, 2, rank_coords);

    cudaFree(0);
    MPI_Barrier(MPI_COMM_WORLD);

    /*setup IB communication infrastructure*/
    if(use_mpi == 0)
    {
        int ret = MP_SUCCESS;
        ret = mp_init (comm2d, peers, peer_count, MP_INIT_DEFAULT, dev_id); 
        if (ret != MP_SUCCESS) {
            fprintf(stderr, "mp_init returned error \n");
            exit(-1);
        }        
    }

    if (min_size < 4*boundary_compute_width) 
         min_size = 4*boundary_compute_width;

    if (comm_rank == 0) { 
        //printf("# Size \t time_elapsed\n");
        printf("time_elapsed\n");
    }

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
    //for (i=100; i<=1000; i+=100) 
    for (i=min_size; i<=max_size; i*=2) 
#endif
    {
        if (i > 1024) {
            iter_count = iter_count_large;
            iter_warmup = iter_warmup_large;
        }
        exchange(comm2d, i, boundary_compute_width, iter_count, iter_warmup);
    }

    if(use_mpi == 0)
        mp_finalize();      

    MPI_Comm_free (&comm2d);
    MPI_Finalize();

    return 0;
}
