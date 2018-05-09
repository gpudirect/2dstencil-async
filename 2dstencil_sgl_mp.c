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

/**TODO: 
 * 1) add sync at the end of each iteration
 * 2) make pack/unpack uniform across the different versions
 */

#include <mpi.h>
#include "validate.h"
#include <mp.h>
#include "pack.h"
#include <string.h>
#include <sys/uio.h>
#include <stdio.h>

#ifdef USE_PROF
#include "prof.h"
#else
struct prof { };
#define PROF(P, H) do { } while(0)
static inline int prof_init(struct prof *p, int unit_scale, int scale_factor, const char* unit_scale_str, int nbins, int merge_bins, const char *tags) {return 0;}
static inline int prof_destroy(struct prof *p) {return 0;}
static inline void prof_dump(struct prof *p) {}
static inline void prof_update(struct prof *p) {}
static inline void prof_enable(struct prof *p) {}
static inline int  prof_enabled(struct prof *p) { return 0; }
static inline void prof_disable(struct prof *p) {}
static inline void prof_reset(struct prof *p) {}
#endif

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
        fprintf(stderr, "[%s:%d] cuda failed with %s \n",   \
         __FILE__, __LINE__,cudaGetErrorString(result));\
        exit(-1);                                       \
    }                                                   \
    assert(cudaSuccess == result);                      \
} while (0)

struct prof prof;
struct prof prof_isend_prepare;
int prof_start = 0;
int prof_idx = 0;

long long int min_size = 8, max_size = 16, default_boundary = 2;
int comm_size, px, py;
int comm_rank, rank_coords[2], rank_base[2];
int left, right, bottom, top, peers[4];
int peer_count, req_per_step; 
int sreq_max_inflight, rreq_max_inflight, steps_per_batch = 4, batches_inflight = 4, buffering = 2, prepost_depth = 4;
int threadsperblock = 512, gridsize = 15;

static inline int intlog2(int x) {
    int result = 0, temp = x;
    while (temp >>= 1) result++;
    return result;
}

int exchange (MPI_Comm comm2d, long long int size, int boundary, int iter_count, int iter_warmup)
{
    long long int i, buf_size, msg_size, msg_size_bytes; 
    float *buf_u = NULL, *buf_v = NULL;
    mp_reg_t reg_u, reg_v;
    struct iovec *siov_left_u, *siov_right_u, *riov_left_u, *riov_right_u;
    struct iovec *siov_left_v, *siov_right_v, *riov_left_v, *riov_right_v;

    void *temp;

    float *compute_buf, *exchange_buf;
    mp_reg_t reg;
    struct iovec *siov_left, *siov_right, *riov_left, *riov_right;

    float *pp_compute_buf, *pp_exchange_buf;
    mp_reg_t pp_reg;
    struct iovec *pp_riov_left, *pp_riov_right;

#ifdef _VALIDATE_
    int x, y;
    float *buf_u_h = NULL, *buf_v_h = NULL;
#endif
    int rreq_idx, sreq_idx, wait_idx, rreq_inflight, sreq_inflight, complete_sync_inflight, complete_sync_idx, complete_sync_wait;
    int sreq_idx_start;
    mp_request_t *sreq = NULL, *rreq = NULL;

    int buf_sync[2];
    mp_reg_t reg_sync;
    mp_request_t *sreq_sync = NULL, *rreq_sync = NULL;
    int sreq_sync_idx;

    cudaEvent_t start_event, stop_event, *sendrecv_sync_event, *interior_sync_event;
    float time_elapsed; 
    long int time_start, time_stop, time_prepost;
    cudaStream_t interior_stream; 
    cudaStream_t boundary_sendrecv_stream;
    int complete_sreq_idx, complete_rreq_idx; 

    long long int boundary_log, size_log, size2_log;
    int prepost_depth_warmup, prepost_depth_iter; 

    prepost_depth_warmup = (prepost_depth < iter_warmup) ? prepost_depth : iter_warmup;
    prepost_depth_iter = (prepost_depth < iter_count) ? prepost_depth : iter_count;

    buf_size = sizeof(float)*(size+2)*(size+2);
    msg_size = size;
    msg_size_bytes = msg_size*sizeof(float);

    /*allocating requests*/	
    sreq_max_inflight = req_per_step*steps_per_batch*batches_inflight; 
    rreq_max_inflight = req_per_step*(steps_per_batch*batches_inflight + prepost_depth); 
    sreq = (mp_request_t *) malloc(sreq_max_inflight*sizeof(mp_request_t));
    rreq = (mp_request_t *) malloc(rreq_max_inflight*sizeof(mp_request_t));
    sreq_sync = (mp_request_t *) malloc(sreq_max_inflight*sizeof(mp_request_t));
    rreq_sync = (mp_request_t *) malloc(rreq_max_inflight*sizeof(mp_request_t));

    sendrecv_sync_event = (cudaEvent_t *) malloc (batches_inflight*steps_per_batch*sizeof(cudaEvent_t));
    interior_sync_event = (cudaEvent_t *) malloc (batches_inflight*steps_per_batch*sizeof(cudaEvent_t));
   
    CUDA_CHECK(cudaMalloc((void **)&buf_u, buf_size));
    CUDA_CHECK(cudaMalloc((void **)&buf_v, buf_size));
    mp_register(buf_u, buf_size, &reg_u);
    mp_register(buf_v, buf_size, &reg_v);
    mp_register(buf_sync, sizeof(int)*2, &reg_sync);

    siov_left_u = malloc(sizeof(struct iovec)*size);
    siov_left_v = malloc(sizeof(struct iovec)*size);
    siov_right_u = malloc(sizeof(struct iovec)*size);
    siov_right_v = malloc(sizeof(struct iovec)*size);
    riov_left_u = malloc(sizeof(struct iovec)*size);
    riov_left_v = malloc(sizeof(struct iovec)*size);
    riov_right_u = malloc(sizeof(struct iovec)*size);
    riov_right_v = malloc(sizeof(struct iovec)*size);
    for (i=0; i<size; i++) { 
	siov_left_u[i].iov_base = (void *)(buf_u + 1 + i*(size+2));
	siov_right_u[i].iov_base = (void *)(buf_u + size + i*(size+2));
	siov_left_v[i].iov_base = (void *)(buf_v + 1 + i*(size+2));
	siov_right_v[i].iov_base = (void *)(buf_v + size + i*(size+2));
	siov_left_u[i].iov_len = siov_right_u[i].iov_len = siov_left_v[i].iov_len = siov_right_v[i].iov_len = sizeof(float);
	riov_left_u[i].iov_base = (void *)(buf_u + i*(size+2));
	riov_right_u[i].iov_base = (void *)(buf_u + (size + 1) + i*(size+2));
	riov_left_v[i].iov_base = (void *)(buf_v + i*(size+2));
	riov_right_v[i].iov_base = (void *)(buf_v + (size+1) + i*(size+2));
	riov_left_u[i].iov_len = riov_right_u[i].iov_len = riov_left_v[i].iov_len = riov_right_v[i].iov_len = sizeof(float);
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

    cudaMemcpy (buf_u, buf_u_h, buf_size, cudaMemcpyHostToDevice);
    cudaMemcpy (buf_v, buf_v_h, buf_size, cudaMemcpyHostToDevice);
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
    siov_left = siov_left_v;
    siov_right = siov_right_v;
    riov_left = riov_left_v;
    riov_right = riov_right_v;
    reg = reg_v;

    pp_compute_buf = buf_u; 
    pp_exchange_buf = buf_v;
    pp_riov_left = riov_left_v;
    pp_riov_right = riov_right_v;
    pp_reg = reg_v;


    rreq_idx = 0;
    sreq_idx = 0;
    complete_sreq_idx = 0;
    complete_rreq_idx = 0;
    rreq_inflight = 0;
    sreq_inflight = 0;
    complete_sync_inflight = 0;
    complete_sync_idx = 0;
    complete_sync_wait = 0;
    sreq_sync_idx = 0;

    int doublebuf_idx;

    fprintf(stderr, "left: %d right: %d top: %d bottom: %d \n", left, right, top, bottom);

    /*warmup steps*/
    for (i=0; i<prepost_depth_warmup; i++) {
        int temp_idx = rreq_idx; 

        /*y dim*/
        if (left != -1) {
            mp_irecvv(pp_riov_left, size, left, &pp_reg,
                     &rreq[rreq_idx]);

            rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
            rreq_inflight++;
        }
        if (right != -1) {
            mp_irecvv(pp_riov_right, size, right, &pp_reg,
                     &rreq[rreq_idx]);

            rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
            rreq_inflight++;
        }
        /*x dim*/
        if (bottom != -1) {
            mp_irecv((void *)(pp_exchange_buf + (size+2)*(size+1) + 1),
                    msg_size_bytes, bottom, &pp_reg, &rreq[rreq_idx]);

            rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
            rreq_inflight++;
        }
        if (top != -1) {
            mp_irecv((void *)(pp_exchange_buf + 1),
                    msg_size_bytes, top, &pp_reg, &rreq[rreq_idx]);

            rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
            rreq_inflight++;
        }

        rreq_idx = temp_idx; 

        /*y dim*/
        if (left != -1) {
	    mp_irecv (buf_sync + 1, sizeof(int), left, &reg, 
		&rreq_sync[rreq_idx]);

            rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
        }
        if (right != -1) {
	    mp_irecv (buf_sync + 1, sizeof(int), right, &reg, 
		&rreq_sync[rreq_idx]);

            rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
        }
        /*x dim*/
        if (bottom != -1) {
	    mp_irecv (buf_sync + 1, sizeof(int), bottom, &reg, 
		&rreq_sync[rreq_idx]);

            rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
        }
        if (top != -1) {
	    mp_irecv (buf_sync + 1, sizeof(int), top, &reg, 
		&rreq_sync[rreq_idx]);

            rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
        }

        temp = (void *)pp_exchange_buf;
        pp_exchange_buf = pp_compute_buf;
        pp_compute_buf = (float *)temp;
        pp_riov_left = (pp_riov_left == riov_left_u) ? riov_left_v : riov_left_u;
        pp_riov_right = (pp_riov_right == riov_right_u) ? riov_right_v: riov_right_u;
        pp_reg = (pp_reg == reg_u) ? reg_v : reg_u;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < iter_warmup; i++) {
	doublebuf_idx = i%buffering;

#ifdef _INTERIOR_FIRST_
	/*launch interior compute*/
	interior_compute (compute_buf, exchange_buf, size, boundary, threadsperblock, gridsize, interior_stream);
#endif

	/*prepare and post requires requests to be contiguous, check if there is a wrap around*/
	int sreq_idx_start = sreq_idx;
        if ((sreq_idx_start + req_per_step) > sreq_max_inflight) {
            fprintf(stderr, "sreq status array looping around in middle of a step: sreq_idx_start: %d sreq_max_inflight: %d \n",
                        sreq_idx_start, sreq_max_inflight);
            exit(-1);
        }

	/*prepare all sends*/
        /*y dim*/
        if (right != -1) {
            mp_sendv_prepare(siov_right, size, right, &reg, 
                    &sreq[sreq_idx]);
            sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
            sreq_inflight++;
        }
        if (left != -1) {
            mp_sendv_prepare(siov_left, size, left, &reg, 
        	    &sreq[sreq_idx]);
            sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
            sreq_inflight++;
        }
        /*x dim*/
        if (top != -1) {
            mp_send_prepare(exchange_buf + (size+2)*size + 1,
                    msg_size_bytes, top, &reg, &sreq[sreq_idx]);
            sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
            sreq_inflight++;
        }
        if (bottom != -1) {
            mp_send_prepare(exchange_buf + (size+2) + 1,
                    msg_size_bytes, bottom, &reg, &sreq[sreq_idx]);
            sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
            sreq_inflight++;
        }

	/*post sends on stream*/
        mp_isend_post_all_on_stream (req_per_step, &sreq[sreq_idx_start], boundary_sendrecv_stream);

#ifndef _INTERIOR_FIRST_
        /*launch interior compute*/
        interior_compute (compute_buf, exchange_buf, size, boundary, threadsperblock, gridsize, interior_stream);
#endif

        /*post wait for recv on the stream*/
        wait_idx = (i*req_per_step)%rreq_max_inflight;
        mp_wait_all_on_stream(req_per_step, &rreq[wait_idx], boundary_sendrecv_stream);

	/*launch boundary computation*/
        boundary_compute (compute_buf, exchange_buf, size, boundary, threadsperblock, gridsize, boundary_sendrecv_stream);

	/*reverse sync using send/recv*/
        sreq_idx_start = sreq_sync_idx; 
        if ((sreq_idx_start + req_per_step) > sreq_max_inflight) {
            fprintf(stderr, "sreq status array looping around in middle of a step: sreq_idx_start: %d sreq_max_inflight: %d \n",
                        sreq_idx_start, sreq_max_inflight);
            exit(-1);
        }

	fprintf(stderr, "left: %d right: %d top: %d bottom: %d \n", left, right, top, bottom);

        if (right != -1) {
		fprintf(stderr, "sending right .... \n"); 
		mp_send_prepare(buf_sync, sizeof(int), right, &reg, 
			&sreq_sync[sreq_sync_idx]);
		sreq_sync_idx = (sreq_sync_idx + 1)%(req_per_step*steps_per_batch*batches_inflight);
	}
        if (left != -1) {
		fprintf(stderr, "sending left .... \n"); 
	 	mp_send_prepare(buf_sync, sizeof(int), left, &reg, 
			&sreq_sync[sreq_sync_idx]);
		sreq_sync_idx = (sreq_sync_idx + 1)%(req_per_step*steps_per_batch*batches_inflight);
	}
        if (top != -1) 	{ 
		fprintf(stderr, "sending top .... \n"); 
		mp_send_prepare(buf_sync, sizeof(int), top, &reg, 
			&sreq_sync[sreq_sync_idx]);
		sreq_sync_idx = (sreq_sync_idx + 1)%(req_per_step*steps_per_batch*batches_inflight);
	}
        if (bottom != -1) { 
		fprintf(stderr, "sending bottom .... \n"); 
		mp_send_prepare(buf_sync, sizeof(int), bottom, &reg, 
			&sreq_sync[sreq_sync_idx]);
		sreq_sync_idx = (sreq_sync_idx + 1)%(req_per_step*steps_per_batch*batches_inflight);
	}

        mp_isend_post_all_on_stream (req_per_step, &sreq_sync[sreq_idx_start], boundary_sendrecv_stream);

        /*post wait for sync recv on the stream*/
        wait_idx = (i*req_per_step)%rreq_max_inflight;
        mp_wait_all_on_stream(req_per_step, &rreq_sync[wait_idx], boundary_sendrecv_stream);

        CUDA_CHECK(cudaEventRecord(sendrecv_sync_event[complete_sync_idx], boundary_sendrecv_stream));
        CUDA_CHECK(cudaEventRecord(interior_sync_event[complete_sync_idx], interior_stream));

        CUDA_CHECK(cudaStreamWaitEvent(boundary_sendrecv_stream, interior_sync_event[complete_sync_idx], 0));
        CUDA_CHECK(cudaStreamWaitEvent(interior_stream, sendrecv_sync_event[complete_sync_idx], 0));

	complete_sync_inflight++;
        complete_sync_idx = (complete_sync_idx + 1)%(batches_inflight*steps_per_batch);

        assert(complete_sync_inflight <= batches_inflight*steps_per_batch);

	/*synchronize on oldest batch*/
        if (complete_sync_inflight == batches_inflight*steps_per_batch) {
            complete_sync_wait = (complete_sync_wait + steps_per_batch - 1)%(batches_inflight*steps_per_batch);

            CUDA_CHECK(cudaEventSynchronize(sendrecv_sync_event[complete_sync_wait]));
            CUDA_CHECK(cudaEventSynchronize(interior_sync_event[complete_sync_wait]));
            complete_sync_wait = (complete_sync_wait + 1)%(batches_inflight*steps_per_batch);
	    complete_sync_inflight = complete_sync_inflight - steps_per_batch; 

            while (rreq_inflight > req_per_step*(steps_per_batch*(batches_inflight - 1) + prepost_depth_warmup)) {
                mp_wait(&rreq[complete_rreq_idx]);
                mp_wait(&rreq_sync[complete_rreq_idx]);
                complete_rreq_idx = (complete_rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight--; 
            }

            while (sreq_inflight > req_per_step*steps_per_batch*(batches_inflight - 1)) {
                mp_wait(&sreq[complete_sreq_idx]);
                mp_wait(&sreq_sync[complete_sreq_idx]);
                complete_sreq_idx = (complete_sreq_idx + 1)%sreq_max_inflight; 
                sreq_inflight--;
            }
	}

        /*prepost recv for a future step*/ 
        if ((i+prepost_depth_warmup) < iter_warmup) {
             int temp_idx = rreq_idx; 
             
             /*y dim*/
             if (left != -1) {
                 mp_irecvv(pp_riov_left, size, left, &pp_reg,
                          &rreq[rreq_idx]);
             
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                 rreq_inflight++;
             }
             if (right != -1) {
                 mp_irecvv(pp_riov_right, size, right, &pp_reg,
                          &rreq[rreq_idx]);
             
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                 rreq_inflight++;
             }
             /*x dim*/
             if (bottom != -1) {
                 mp_irecv((void *)(pp_exchange_buf + (size+2)*(size+1) + 1),
                         msg_size_bytes, bottom, &pp_reg, &rreq[rreq_idx]);
             
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                 rreq_inflight++;
             }
             if (top != -1) {
                 mp_irecv((void *)(pp_exchange_buf + 1),
                         msg_size_bytes, top, &pp_reg, &rreq[rreq_idx]);
             
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                 rreq_inflight++;
             }
             
             rreq_idx = temp_idx; 
             
             /*y dim*/
             if (left != -1) {
                 mp_irecv (buf_sync + 1, sizeof(int), left, &reg, 
             	&rreq_sync[rreq_idx]);
             
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
             }
             if (right != -1) {
                 mp_irecv (buf_sync + 1, sizeof(int), right, &reg, 
             	&rreq_sync[rreq_idx]);
             
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
             }
             /*x dim*/
             if (bottom != -1) {
                 mp_irecv (buf_sync + 1, sizeof(int), bottom, &reg, 
             	&rreq_sync[rreq_idx]);
             
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
             }
             if (top != -1) {
                 mp_irecv (buf_sync + 1, sizeof(int), top, &reg, 
             	&rreq_sync[rreq_idx]);
             
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
             }

             temp = (void *)pp_exchange_buf;
             pp_exchange_buf = pp_compute_buf;
             pp_compute_buf = (float *)temp;
             pp_riov_left = (pp_riov_left == riov_left_u) ? riov_left_v : riov_left_u;
             pp_riov_right = (pp_riov_right == riov_right_u) ? riov_right_v: riov_right_u;
             pp_reg = (pp_reg == reg_u) ? reg_v : reg_u;
	}

	/*if its the last iteration, synchronize on all batches*/
        if (i == (iter_warmup - 1)) {
            if (complete_sync_inflight > 0) {
                complete_sync_wait = (complete_sync_wait + complete_sync_inflight - 1)%(batches_inflight*steps_per_batch);
                CUDA_CHECK(cudaEventSynchronize(sendrecv_sync_event[complete_sync_wait]));
                CUDA_CHECK(cudaEventSynchronize(interior_sync_event[complete_sync_wait]));

                complete_sync_inflight = 0;
            }
  
            while (rreq_inflight > 0) {
                mp_wait(&rreq[complete_rreq_idx]);
                mp_wait(&rreq_sync[complete_rreq_idx]);
                complete_rreq_idx = (complete_rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight--;
            }
  
            while (sreq_inflight > 0) { 
                mp_wait(&sreq[complete_sreq_idx]);
                mp_wait(&sreq_sync[complete_sreq_idx]);
                complete_sreq_idx = (complete_sreq_idx + 1)%sreq_max_inflight;
                sreq_inflight--;
            }
	}

	/*interchange the compute and communication buffers*/
        temp = (void *)exchange_buf; 
	exchange_buf = compute_buf; 
	compute_buf = (float *)temp;
        siov_left = (siov_left == siov_left_u) ? siov_left_v : siov_left_u; 
    	siov_right = (siov_right == siov_right_u) ? siov_right_v : siov_right_u;
    	riov_left = (riov_left == riov_left_u) ? riov_left_v : riov_left_u; 
        riov_right = (riov_right == riov_right_u) ? riov_right_v: riov_right_u;
    	reg = (reg == reg_u) ? reg_v : reg_u;
    }

#ifdef _REPEAT_TIMED_LOOP_
    int reps_count = 10, rep;
    if (getenv("REPS_COUNT") != NULL) { 
	reps_count = atoi(getenv("REPS_COUNT"));
    }
    for (rep = 0; rep < reps_count; rep++) { 
#endif

    CUDA_CHECK(cudaDeviceSynchronize());

    rreq_idx = 0;
    sreq_idx = 0;
    complete_sreq_idx = 0;
    complete_rreq_idx = 0;
    rreq_inflight = 0;
    sreq_inflight = 0;
    complete_sync_inflight = 0;
    complete_sync_idx = 0;
    complete_sync_wait = 0;
    sreq_sync_idx = 0;

    MPI_Barrier(MPI_COMM_WORLD);

    /*timed iterations*/
    time_start = cycles_to_ns(get_cycles());
#ifndef _FREE_NETWORK_
    for (i=0; i<prepost_depth_iter; i++) {
        int temp_idx = rreq_idx; 

        /*y dim*/
        if (left != -1) {
            mp_irecvv(pp_riov_left, size, left, &pp_reg,
                     &rreq[rreq_idx]);

            rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
            rreq_inflight++;
        }
        if (right != -1) {
            mp_irecvv(pp_riov_right, size, right, &pp_reg,
                     &rreq[rreq_idx]);

            rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
            rreq_inflight++;
        }
        /*x dim*/
        if (bottom != -1) {
            mp_irecv((void *)(pp_exchange_buf + (size+2)*(size+1) + 1),
                    msg_size_bytes, bottom, &pp_reg, &rreq[rreq_idx]);

            rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
            rreq_inflight++;
        }
        if (top != -1) {
            mp_irecv((void *)(pp_exchange_buf + 1),
                    msg_size_bytes, top, &pp_reg, &rreq[rreq_idx]);

            rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
            rreq_inflight++;
        }

        rreq_idx = temp_idx; 

        /*y dim*/
        if (left != -1) {
	    mp_irecv (buf_sync + 1, sizeof(int), left, &reg, 
		&rreq_sync[rreq_idx]);

            rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
        }
        if (right != -1) {
	    mp_irecv (buf_sync + 1, sizeof(int), right, &reg, 
		&rreq_sync[rreq_idx]);

            rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
        }
        /*x dim*/
        if (bottom != -1) {
	    mp_irecv (buf_sync + 1, sizeof(int), bottom, &reg, 
		&rreq_sync[rreq_idx]);

            rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
        }
        if (top != -1) {
	    mp_irecv (buf_sync + 1, sizeof(int), top, &reg, 
		&rreq_sync[rreq_idx]);

            rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
        }

        temp = (void *)pp_exchange_buf;
        pp_exchange_buf = pp_compute_buf;
        pp_compute_buf = (float *)temp;
        pp_riov_left = (pp_riov_left == riov_left_u) ? riov_left_v : riov_left_u;
        pp_riov_right = (pp_riov_right == riov_right_u) ? riov_right_v: riov_right_u;
        pp_reg = (pp_reg == reg_u) ? reg_v : reg_u;
    }
#endif

    time_stop = cycles_to_ns(get_cycles());
    time_prepost = (time_stop - time_start);

#if defined (_ENABLE_DRPROF_)
    const char *tags = "interior|pack|prepsend|postsend|postwait|unpack|boundary|postsync|waitsync|postrecv";
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

#ifdef _INTERIOR_FIRST_
#ifndef _FREE_INTERIOR_COMPUTE_
	/*launch interior compute*/
	interior_compute (compute_buf, exchange_buf, size, boundary, threadsperblock, gridsize, interior_stream);
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif
#endif

        /*post all sends*/ 
#ifndef _FREE_NETWORK_
	/*prepare and post requires requests to be contiguous, check if there is a wrap around*/
	sreq_idx_start = sreq_idx;
	if ((sreq_idx_start + req_per_step) > sreq_max_inflight) {
	    fprintf(stderr, "sreq status array looping around in middle of a step: sreq_idx_start: %d sreq_max_inflight: %d \n", 
			sreq_idx_start, sreq_max_inflight);
	    exit(-1);
	}

	/*prepare all sends*/
        /*y dim*/
        if (right != -1) {
            mp_sendv_prepare(siov_right, size, right, &reg,
                    &sreq[sreq_idx]);
            sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
            sreq_inflight++;
        }
        if (left != -1) {
            mp_sendv_prepare(siov_left, size, left, &reg,
                    &sreq[sreq_idx]);
            sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
            sreq_inflight++;
        }
        /*x dim*/
        if (top != -1) {
            mp_send_prepare(exchange_buf + (size+2)*size + 1,
                    msg_size_bytes, top, &reg, &sreq[sreq_idx]);
            sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
            sreq_inflight++;
        }
        if (bottom != -1) {
            mp_send_prepare(exchange_buf + (size+2) + 1,
                    msg_size_bytes, bottom, &reg, &sreq[sreq_idx]);
            sreq_idx = (sreq_idx + 1)%sreq_max_inflight;
            sreq_inflight++;
        }
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif

#ifndef _FREE_NETWORK_
	/*post sends on stream*/
	mp_isend_post_all_on_stream (req_per_step, &sreq[sreq_idx_start], boundary_sendrecv_stream);
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif

#ifndef _INTERIOR_FIRST_
#ifndef _FREE_INTERIOR_COMPUTE_
        /*launch interior compute*/
        interior_compute (compute_buf, exchange_buf, size, boundary, threadsperblock, gridsize, interior_stream);
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif
#endif

#ifndef _FREE_NETWORK_
	/*post wait for recv on the stream*/
        wait_idx = (i*req_per_step)%rreq_max_inflight;
        mp_wait_all_on_stream(req_per_step, &rreq[wait_idx], boundary_sendrecv_stream);
#endif

        cudaStreamSynchronize(boundary_sendrecv_stream);

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

#ifndef _FREE_NETWORK_
        sreq_idx_start = sreq_sync_idx;
        if ((sreq_idx_start + req_per_step) > sreq_max_inflight) {
            fprintf(stderr, "sreq status array looping around in middle of a step: sreq_idx_start: %d sreq_max_inflight: %d \n",
                        sreq_idx_start, sreq_max_inflight);
            exit(-1);
        }

        if (right != -1) {
		fprintf(stderr, "sending right .... %d \n", sreq_sync_idx); 
		mp_send_prepare(buf_sync, sizeof(int), right, &reg, 
			&sreq_sync[sreq_sync_idx]);
		sreq_sync_idx = (sreq_sync_idx + 1)%(req_per_step*steps_per_batch*batches_inflight);
	}
        if (left != -1) { 
		fprintf(stderr, "sending left .... idx: %d \n", sreq_sync_idx); 
	 	mp_send_prepare(buf_sync, sizeof(int), left, &reg, 
			&sreq_sync[sreq_sync_idx]);
		sreq_sync_idx = (sreq_sync_idx + 1)%(req_per_step*steps_per_batch*batches_inflight);
	}
        if (top != -1) 	{ 
		fprintf(stderr, "sending top .... %d \n", sreq_sync_idx); 
		mp_send_prepare(buf_sync, sizeof(int), top, &reg, 
			&sreq_sync[sreq_sync_idx]);
		sreq_sync_idx = (sreq_sync_idx + 1)%(req_per_step*steps_per_batch*batches_inflight);
	}
        if (bottom != -1) { 
		fprintf(stderr, "sending bottom .... %d \n", sreq_sync_idx); 
		mp_send_prepare(buf_sync, sizeof(int), bottom, &reg, 
			&sreq_sync[sreq_sync_idx]);
		sreq_sync_idx = (sreq_sync_idx + 1)%(req_per_step*steps_per_batch*batches_inflight);
	}

        mp_isend_post_all_on_stream (req_per_step, &sreq_sync[sreq_idx_start], boundary_sendrecv_stream);

	/*post wait for sync recv on the stream*/
        wait_idx = (i*req_per_step)%rreq_max_inflight; 
        mp_wait_all_on_stream(req_per_step, &rreq_sync[wait_idx], boundary_sendrecv_stream);
#endif

        cudaStreamSynchronize(boundary_sendrecv_stream);

#ifndef _FREE_SYNC_
        CUDA_CHECK(cudaEventRecord(sendrecv_sync_event[complete_sync_idx], boundary_sendrecv_stream));
        CUDA_CHECK(cudaEventRecord(interior_sync_event[complete_sync_idx], interior_stream));

        CUDA_CHECK(cudaStreamWaitEvent(boundary_sendrecv_stream, interior_sync_event[complete_sync_idx], 0));
        CUDA_CHECK(cudaStreamWaitEvent(interior_stream, sendrecv_sync_event[complete_sync_idx], 0));
#endif

	complete_sync_inflight++;
        complete_sync_idx = (complete_sync_idx + 1)%(batches_inflight*steps_per_batch);

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif

        assert(complete_sync_inflight <= batches_inflight*steps_per_batch);
	/*synchronize on oldest batch*/
        if (complete_sync_inflight == batches_inflight*steps_per_batch) {

            complete_sync_wait = (complete_sync_wait + steps_per_batch - 1)%(batches_inflight*steps_per_batch);

	    fprintf(stderr, "synchronizing on event \n");
#ifndef _FREE_SYNC_
            CUDA_CHECK(cudaEventSynchronize(sendrecv_sync_event[complete_sync_wait]));
            CUDA_CHECK(cudaEventSynchronize(interior_sync_event[complete_sync_wait]));
#endif
            complete_sync_wait = (complete_sync_wait + 1)%(batches_inflight*steps_per_batch);
	    complete_sync_inflight = complete_sync_inflight - steps_per_batch; 

#ifndef _FREE_NETWORK_
            while (rreq_inflight > req_per_step*(steps_per_batch*(batches_inflight - 1) + prepost_depth_iter)) {
                mp_wait(&rreq[complete_rreq_idx]);
                mp_wait(&rreq_sync[complete_rreq_idx]);
                complete_rreq_idx = (complete_rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight--; 
            }

            while (sreq_inflight > req_per_step*steps_per_batch*(batches_inflight - 1)) {
                mp_wait(&sreq[complete_sreq_idx]);
                mp_wait(&sreq_sync[complete_sreq_idx]);
                complete_sreq_idx = (complete_sreq_idx + 1)%sreq_max_inflight; 
                sreq_inflight--;
            }
#endif
	}

        /*prepost recv for a future step*/ 
#ifndef _FREE_NETWORK_
         if ((i+prepost_depth_iter) < iter_count) {
             int temp_idx = rreq_idx; 
             
             /*y dim*/
             if (left != -1) {
                 mp_irecvv(pp_riov_left, size, left, &pp_reg,
                          &rreq[rreq_idx]);
             
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                 rreq_inflight++;
             }
             if (right != -1) {
                 mp_irecvv(pp_riov_right, size, right, &pp_reg,
                          &rreq[rreq_idx]);
             
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                 rreq_inflight++;
             }
             /*x dim*/
             if (bottom != -1) {
                 mp_irecv((void *)(pp_exchange_buf + (size+2)*(size+1) + 1),
                         msg_size_bytes, bottom, &pp_reg, &rreq[rreq_idx]);
             
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                 rreq_inflight++;
             }
             if (top != -1) {
                 mp_irecv((void *)(pp_exchange_buf + 1),
                         msg_size_bytes, top, &pp_reg, &rreq[rreq_idx]);
             
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
                 rreq_inflight++;
             }
             
             rreq_idx = temp_idx; 
             
             /*y dim*/
             if (left != -1) {
                 mp_irecv (buf_sync + 1, sizeof(int), left, &reg, 
             	&rreq_sync[rreq_idx]);
             
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
             }
             if (right != -1) {
                 mp_irecv (buf_sync + 1, sizeof(int), right, &reg, 
             	&rreq_sync[rreq_idx]);
             
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
             }
             /*x dim*/
             if (bottom != -1) {
                 mp_irecv (buf_sync + 1, sizeof(int), bottom, &reg, 
             	&rreq_sync[rreq_idx]);
             
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
             }
             if (top != -1) {
                 mp_irecv (buf_sync + 1, sizeof(int), top, &reg, 
             	&rreq_sync[rreq_idx]);
             
                 rreq_idx = (rreq_idx + 1)%rreq_max_inflight;
             }

             temp = (void *)pp_exchange_buf;
             pp_exchange_buf = pp_compute_buf;
             pp_compute_buf = (float *)temp;
             pp_riov_left = (pp_riov_left == riov_left_u) ? riov_left_v : riov_left_u;
             pp_riov_right = (pp_riov_right == riov_right_u) ? riov_right_v: riov_right_u;
             pp_reg = (pp_reg == reg_u) ? reg_v : reg_u;
	}
#endif 

	/*if its the last iteration, synchronize on all batches*/
        if (i == (iter_count - 1)) {
            if (complete_sync_inflight > 0) {
                complete_sync_wait = (complete_sync_wait + complete_sync_inflight - 1)%(batches_inflight*steps_per_batch);
#ifndef _FREE_SYNC_
                CUDA_CHECK(cudaEventSynchronize(sendrecv_sync_event[complete_sync_wait]));
                CUDA_CHECK(cudaEventSynchronize(interior_sync_event[complete_sync_wait]));
#endif

                complete_sync_inflight = 0;
            }

 #ifndef _FREE_NETWORK_
            while (rreq_inflight > 0) {
                mp_wait(&rreq[complete_rreq_idx]);
                mp_wait(&rreq_sync[complete_rreq_idx]);
                complete_rreq_idx = (complete_rreq_idx + 1)%rreq_max_inflight;
                rreq_inflight--;
            }
  
            while (sreq_inflight > 0) { 
                mp_wait(&sreq[complete_sreq_idx]);
                mp_wait(&sreq_sync[complete_sreq_idx]);
                complete_sreq_idx = (complete_sreq_idx + 1)%sreq_max_inflight;
                sreq_inflight--;
            }
#endif
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
        siov_left = (siov_left == siov_left_u) ? siov_left_v : siov_left_u;
        siov_right = (siov_right == siov_right_u) ? siov_right_v : siov_right_u;
        riov_left = (riov_left == riov_left_u) ? riov_left_v : riov_left_u;
        riov_right = (riov_right == riov_right_u) ? riov_right_v: riov_right_u;
        reg = (reg == reg_u) ? reg_v : reg_u;
    }

    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&time_elapsed, start_event, stop_event));
    if (comm_rank == 0) {
#ifdef _REPEAT_TIMED_LOOP_
        fprintf(stdout, "%3lld %3lld %8.2lf usec \n", size, rep, (time_elapsed*1e3 + (time_prepost/1000))/iter_count);
#else
        fprintf(stdout, "%3lld %8.2lf usec \n", size, (time_elapsed*1e3 + (time_prepost/1000))/iter_count);
#endif

#if defined (_ENABLE_DRPROF_)
	fprintf(stdout, "Prepost depth: %d latency: %ld nsec \n", prepost_depth_iter, time_prepost);
        prof_dump(&prof);
#if defined (_ENABLE_DRPROF_ISEND_)
        prof_dump(&prof_isend_prepare);
#endif
#endif
    }

#ifdef _REPEAT_TIMED_LOOP_
    }

#ifdef _VALIDATE_
    iter_count = iter_count*reps_count;
#endif
#endif

#ifdef _VALIDATE_
    emulate_on_host (buf_u_h, buf_v_h, size, boundary, 1/*ghost*/, 
            comm_rank, left, right, bottom, top, (iter_count + iter_warmup), comm2d);

    cudaMemcpy (buf_u_h, buf_u, buf_size, cudaMemcpyDeviceToHost);
    cudaMemcpy (buf_v_h, buf_v, buf_size, cudaMemcpyDeviceToHost);

    validate (buf_u_h, buf_v_h, size, 1);
#endif

    free(siov_left_u);
    free(siov_left_v); 
    free(siov_right_u);
    free(siov_right_v);
    free(riov_left_u); 
    free(riov_left_v); 
    free(riov_right_u);
    free(riov_right_v);
    mp_deregister(&reg_u);
    mp_deregister(&reg_v);

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaStreamDestroy(interior_stream));
    CUDA_CHECK(cudaStreamDestroy(boundary_sendrecv_stream));
    for (i=0; i<batches_inflight*steps_per_batch; i++) {
        CUDA_CHECK(cudaEventDestroy(sendrecv_sync_event[i]));
        CUDA_CHECK(cudaEventDestroy(interior_sync_event[i]));
    }

    free(sreq);
    free(rreq);

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
	prepost_depth = steps_per_batch;
    }
    if (getenv("BATCHES_INFLIGHT") != NULL) {
        batches_inflight = atoi(getenv("BATCHES_INFLIGHT"));
    }
    if (getenv("BUFFERING") != NULL) {
        buffering = atoi(getenv("BUFFERING"));
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

    cudaSetDevice(dev_id);
    fprintf(stderr, "[%d] using GPU device: %d \n", comm_rank, dev_id);  

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
    fprintf(stderr, "[%d] left: %d right: %d top: %d bottom: %d iters_per_batch: %d batches_inflight: %d buffering: %d iter_count: %d iter_warmup: %d\n", comm_rank, left, right, top, bottom, steps_per_batch, batches_inflight, buffering, iter_count, iter_warmup);

    if (left != -1) {
	req_per_step++; 
        peers[peer_count] = left;
        peer_count++;
    }
    if (right != -1) {
	req_per_step++; 
	if (right != left) { 
            peers[peer_count] = right;
            peer_count++;
	}
    }
    if (top != -1) { 
	req_per_step++; 
        peers[peer_count] = top;
        peer_count++;
    }
    if (bottom != -1) { 
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
    int ret = MP_SUCCESS;
    ret = mp_init (comm2d, peers, peer_count, dev_id, MP_INIT_DEFAULT); 
    if (ret != MP_SUCCESS) {
        fprintf(stderr, "mp_init returned error \n");
        exit(-1);
    }

    if (min_size < 4*boundary_compute_width) 
         min_size = 4*boundary_compute_width;

    if (comm_rank == 0) { 
        fprintf(stderr, "Size \t time_elapsed \n");
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

    MPI_Comm_free (&comm2d);
    MPI_Finalize();

    return 0;
}
