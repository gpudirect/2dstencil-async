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

#include <mpi.h>
#include <string.h>
#include <mpi.h>

#include "common.h"
#include "validate_strong.h"
#include "ib.h"
#include "pack_strong.h"

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

struct prof prof;
int prof_idx = 0;

/*    x         
 *    ^      
 *    -    
 *    -  
 *    -
 *  0 --------> y
 * */

long long int min_size = 8, max_size = 8*1024, default_boundary = 2;
int comm_size, px, py;
int comm_rank, rank_coords[2], rank_base[2];
int left, right, bottom, top, peers[4];
int threadsperblock = 512, gridsize = 15;
int peer_count = 0;
int buffering = 2, prepost_depth_default = 4;
int max_comm_size = 65536;

int exchange (MPI_Comm comm2d, int npx, int npy, long long int sizex, long long int sizey, int boundary, int iter_count, int iter_warmup)
{
    long long int i, x, buf_size, msg_size, msg_size_bytes, packbuf_disp; 
    float *buf_u = NULL, *buf_v = NULL, **packbuf = NULL, **unpackbuf = NULL;
    int size, faces = 4;
    ib_reg_t **reg_pack, **reg_unpack;
#ifdef _VALIDATE_
    int y;
    float *buf_u_h = NULL, *buf_v_h = NULL;
#endif
    int rreq_idx, sreq_idx; 
    ib_status_t *sreq_status = NULL;
    ib_status_t *rreq_status = NULL;
    cudaEvent_t start_event, stop_event;
    float time_elapsed;
    long int time_start, time_stop, time_prepost;
    cudaStream_t interior_stream, boundary_stream; 
    int complete_idx, prepost_depth;

    long long int boundary_log, sizex_log, sizey_log, sizexy_log;
    float *compute_buf, *exchange_buf, *temp;

    size = sizex > sizey ? sizex : sizey;
    buf_size = sizeof(float)*(sizex+2)*(sizey+2);
    msg_size = size;
    msg_size_bytes = size*sizeof(float);

    /*allocating requests*/	
    sreq_status = (ib_status_t *) malloc(peer_count*sizeof(ib_status_t));
    rreq_status = (ib_status_t *) malloc(peer_count*sizeof(ib_status_t)*prepost_depth_default);

    CUDA_CHECK(cudaMalloc((void **)&buf_u, buf_size));
    CUDA_CHECK(cudaMalloc((void **)&buf_v, buf_size));

    packbuf = (float **) malloc(sizeof(float *)*buffering);
    unpackbuf = (float **) malloc(sizeof(float *)*buffering);
    reg_pack = (ib_reg_t **) malloc (sizeof(ib_reg_t *)*buffering); 
    reg_unpack = (ib_reg_t **) malloc (sizeof(ib_reg_t *)*buffering); 

    for (x=0; x<buffering; x++) {  
        CUDA_CHECK(cudaMalloc((void **)&packbuf[x], msg_size_bytes*faces));
        CUDA_CHECK(cudaMalloc((void **)&unpackbuf[x], msg_size_bytes*faces));

        CUDA_CHECK(cudaMemset(packbuf[x], 0, msg_size_bytes*faces));
        CUDA_CHECK(cudaMemset(unpackbuf[x], 0, msg_size_bytes*faces));

        reg_pack[x] = ib_register(packbuf[x], msg_size_bytes*faces); 
        reg_unpack[x] = ib_register(unpackbuf[x], msg_size_bytes*faces); 
    }

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

#ifndef _USE_NONBLOCKING_STREAMS_
    CUDA_CHECK(cudaStreamCreate(&interior_stream));
    CUDA_CHECK(cudaStreamCreate(&boundary_stream));
#else
    CUDA_CHECK(cudaStreamCreateWithFlags(&interior_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&boundary_stream, cudaStreamNonBlocking));
#endif

    CUDA_CHECK(cudaDeviceSynchronize());

    compute_buf = buf_u; 
    exchange_buf = buf_v;

    /*warmup steps*/
    prepost_depth = (prepost_depth_default < iter_warmup) ? prepost_depth_default : iter_warmup;
    for (i=0; i<prepost_depth; i++) {
	int stage_buf_idx = i%buffering;
        int rreq_idx=(i%prepost_depth)*peer_count;
	int packbuf_disp = 0;

        /*y dim*/
        if (left != MPI_PROC_NULL) {
            ib_irecv((void *)(unpackbuf[stage_buf_idx] + packbuf_disp), msg_size_bytes, left,
                                reg_unpack[stage_buf_idx], &rreq_status[rreq_idx]);
            rreq_idx++;
        }
        packbuf_disp += msg_size;
        if (right != MPI_PROC_NULL) {
            ib_irecv((void *)(unpackbuf[stage_buf_idx] + packbuf_disp),
                    msg_size_bytes, right, reg_unpack[stage_buf_idx],
                    &rreq_status[rreq_idx]);
            rreq_idx++;
        }
        packbuf_disp += msg_size;
        /*x dim*/
        if (bottom != MPI_PROC_NULL) {
            ib_irecv((void *)(unpackbuf[stage_buf_idx] + packbuf_disp),
                    msg_size_bytes, bottom, reg_unpack[stage_buf_idx],
                    &rreq_status[rreq_idx]);
            rreq_idx++;
        }
        packbuf_disp += msg_size;
        if (top != MPI_PROC_NULL) {
            ib_irecv((void *)(unpackbuf[stage_buf_idx] + packbuf_disp),
                    msg_size_bytes, top, reg_unpack[stage_buf_idx],
                    &rreq_status[rreq_idx]);
            rreq_idx++;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < iter_warmup; i++) {
        int stage_buf_idx = i%buffering;
         
        sreq_idx = 0;

#ifdef _INTERIOR_FIRST_
	/*launch interior compute*/
        interior_compute (compute_buf, exchange_buf, sizex, sizey, boundary, threadsperblock, gridsize, interior_stream);
#endif

	packbuf_disp = 0;
        sreq_idx = 0;

	/*pack data*/
   	boundary_pack (packbuf[stage_buf_idx], exchange_buf, sizex, sizey, threadsperblock, boundary_stream);

	CUDA_CHECK(cudaStreamSynchronize(boundary_stream));

        /*post all sends*/ 
        packbuf_disp = 0;

        /*y dim*/
        if (left != MPI_PROC_NULL) {
            ib_isend((void *)(packbuf[stage_buf_idx] + packbuf_disp),
                    msg_size_bytes, left, reg_pack[stage_buf_idx], 
		    &sreq_status[sreq_idx]);
            sreq_idx++;
        }
        packbuf_disp += msg_size;
        if (right != MPI_PROC_NULL) {
            ib_isend((void *)(packbuf[stage_buf_idx] + packbuf_disp),
                    msg_size_bytes, right, reg_pack[stage_buf_idx],
                    &sreq_status[sreq_idx]);
            sreq_idx++;
        }
        packbuf_disp += msg_size;
        /*x dim*/
        if (bottom != MPI_PROC_NULL) {
            ib_isend((void *)(packbuf[stage_buf_idx] + packbuf_disp),
                    msg_size_bytes, bottom, reg_pack[stage_buf_idx],
                    &sreq_status[sreq_idx]);
            sreq_idx++;
        }
        packbuf_disp += msg_size;
        if (top != MPI_PROC_NULL) {
            ib_isend((void *)(packbuf[stage_buf_idx] + packbuf_disp),
                    msg_size_bytes, top, reg_pack[stage_buf_idx],
                    &sreq_status[sreq_idx]);
            sreq_idx++;
        }
        packbuf_disp += msg_size;

#ifndef _INTERIOR_FIRST_
        /*launch interior compute*/
        interior_compute (compute_buf, exchange_buf, sizex, sizey, boundary, threadsperblock, gridsize, interior_stream);
#endif

	complete_idx = 0;
	rreq_idx = (i%buffering)*peer_count + complete_idx;
        while (complete_idx < peer_count) {
	    rreq_idx = (i%prepost_depth)*peer_count + complete_idx;
            if (rreq_status[rreq_idx].status == PENDING) {
                ib_progress_recv();
                continue;
            }
            complete_idx++; 
        }

        complete_idx = 0; 
        while (complete_idx < sreq_idx) { 
            if (sreq_status[complete_idx].status == PENDING) {
                ib_progress_send();
                continue;
            } 
            complete_idx++; 
        }

        /*unpack data*/
        boundary_unpack (exchange_buf, unpackbuf[stage_buf_idx], sizex, sizey, threadsperblock, boundary_stream);

	/*launch boundary computation*/
        boundary_compute (compute_buf, exchange_buf, sizex, sizey, boundary, threadsperblock, gridsize, boundary_stream);

	CUDA_CHECK(cudaDeviceSynchronize());

        if (i+prepost_depth < iter_warmup) {
	    int stage_buf_prepost_idx = (i+prepost_depth)%buffering;
	    int rreq_prepost_idx = ((i+prepost_depth)%prepost_depth)*peer_count;
	    packbuf_disp = 0;	

            /*y dim*/
            if (left != MPI_PROC_NULL) { 
                ib_irecv((void *)(unpackbuf[stage_buf_prepost_idx] + packbuf_disp), 
				msg_size_bytes, left, reg_unpack[stage_buf_prepost_idx], 
				&rreq_status[rreq_prepost_idx]);
                rreq_prepost_idx++;
            }
            packbuf_disp += msg_size;
            if (right != MPI_PROC_NULL) { 
                ib_irecv((void *)(unpackbuf[stage_buf_prepost_idx] + packbuf_disp),
                        	msg_size_bytes, right, reg_unpack[stage_buf_prepost_idx], 
                  	    	&rreq_status[rreq_prepost_idx]);
                rreq_prepost_idx++;
            }
            packbuf_disp += msg_size;
            /*x dim*/
            if (bottom != MPI_PROC_NULL) {
                ib_irecv((void *)(unpackbuf[stage_buf_prepost_idx] + packbuf_disp),
                        msg_size_bytes, bottom, reg_unpack[stage_buf_prepost_idx],
                        &rreq_status[rreq_prepost_idx]);
                rreq_prepost_idx++;
            }
            packbuf_disp += msg_size;
            if (top != MPI_PROC_NULL) {
                ib_irecv((void *)(unpackbuf[stage_buf_prepost_idx] + packbuf_disp),
                        msg_size_bytes, top, reg_unpack[stage_buf_prepost_idx],
                        &rreq_status[rreq_prepost_idx]);
                rreq_prepost_idx++;
            }
	}

	/*intercahnge the compute and communication buffers*/
        temp = exchange_buf; 
	exchange_buf = compute_buf; 
	compute_buf = temp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    time_start = cycles_to_ns(get_cycles());
    /*post receives until prepost depth*/
#ifndef _FREE_NETWORK_
    prepost_depth = (prepost_depth_default < iter_count) ? prepost_depth_default : iter_count;
    for (i=0; i<prepost_depth; i++) {
	int stage_buf_idx = i%buffering;
        int rreq_idx=(i%prepost_depth)*peer_count;
	int packbuf_disp = 0;

        /*y dim*/
        if (left != MPI_PROC_NULL) {
            ib_irecv((void *)(unpackbuf[stage_buf_idx] + packbuf_disp), msg_size_bytes, left,
                                reg_unpack[stage_buf_idx], &rreq_status[rreq_idx]);
            rreq_idx++;
        }
        packbuf_disp += msg_size;
        if (right != MPI_PROC_NULL) {
            ib_irecv((void *)(unpackbuf[stage_buf_idx] + packbuf_disp),
                    msg_size_bytes, right, reg_unpack[stage_buf_idx],
                    &rreq_status[rreq_idx]);
            rreq_idx++;
        }
        packbuf_disp += msg_size;
        /*x dim*/
        if (bottom != MPI_PROC_NULL) {
            ib_irecv((void *)(unpackbuf[stage_buf_idx] + packbuf_disp),
                    msg_size_bytes, bottom, reg_unpack[stage_buf_idx],
                    &rreq_status[rreq_idx]);
            rreq_idx++;
        }
        packbuf_disp += msg_size;
        if (top != MPI_PROC_NULL) {
            ib_irecv((void *)(unpackbuf[stage_buf_idx] + packbuf_disp),
                    msg_size_bytes, top, reg_unpack[stage_buf_idx],
                    &rreq_status[rreq_idx]);
            rreq_idx++;
        }
    }
#endif
    time_stop = cycles_to_ns(get_cycles());
    time_prepost = (time_stop - time_start);

#if defined (_ENABLE_DRPROF_)
    const char *tags = "interior|pack|packsync|postsend|wait|unpack|boundary|gpusync|postrecv";
    if (prof_init(&prof, 1000,  1000, "1us", 50, 1, tags)) {
        fprintf(stderr, "error in prof_init init.\n");
        exit(-1);
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CHECK(cudaEventRecord(start_event, 0));

    for (i = 0; i < iter_count; i++) {
        int stage_buf_idx = i%buffering;
         
        sreq_idx = 0;

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif

#ifdef _INTERIOR_FIRST_
	/*exchange boundary data*/	
#ifndef _FREE_INTERIOR_COMPUTE_
	/*launch interior compute*/
        interior_compute (compute_buf, exchange_buf, sizex, sizey, boundary, threadsperblock, gridsize, interior_stream);
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif
#endif

	packbuf_disp = 0;
        sreq_idx = 0;

#ifndef _FREE_PACK_
	/*pack data*/
   	boundary_pack (packbuf[stage_buf_idx], exchange_buf, sizex, sizey, threadsperblock, boundary_stream);
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif

#ifndef _FREE_SYNC_
	CUDA_CHECK(cudaStreamSynchronize(boundary_stream));
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif

        /*post all sends*/ 
        packbuf_disp = 0;

#ifndef _FREE_NETWORK_
        /*y dim*/
        if (left != MPI_PROC_NULL) {
            ib_isend((void *)(packbuf[stage_buf_idx] + packbuf_disp),
                    msg_size_bytes, left, reg_pack[stage_buf_idx], 
		    &sreq_status[sreq_idx]);
            sreq_idx++;
        }
        packbuf_disp += msg_size;
        if (right != MPI_PROC_NULL) {
            ib_isend((void *)(packbuf[stage_buf_idx] + packbuf_disp),
                    msg_size_bytes, right, reg_pack[stage_buf_idx],
                    &sreq_status[sreq_idx]);
            sreq_idx++;
        }
        packbuf_disp += msg_size;
        /*x dim*/
        if (bottom != MPI_PROC_NULL) {
            ib_isend((void *)(packbuf[stage_buf_idx] + packbuf_disp),
                    msg_size_bytes, bottom, reg_pack[stage_buf_idx],
                    &sreq_status[sreq_idx]);
            sreq_idx++;
        }
        packbuf_disp += msg_size;
        if (top != MPI_PROC_NULL) {
            ib_isend((void *)(packbuf[stage_buf_idx] + packbuf_disp),
                    msg_size_bytes, top, reg_pack[stage_buf_idx],
                    &sreq_status[sreq_idx]);
            sreq_idx++;
        }
        packbuf_disp += msg_size;
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif

#ifndef _INTERIOR_FIRST_
        /*exchange boundary data*/
#ifndef _FREE_INTERIOR_COMPUTE_
        /*launch interior compute*/
        interior_compute (compute_buf, exchange_buf, sizex, sizey, boundary, threadsperblock, gridsize, interior_stream);
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif
#endif

#ifndef _FREE_NETWORK_
	complete_idx = 0;
	rreq_idx = (i%buffering)*peer_count + complete_idx;
        while (complete_idx < peer_count) {
	    rreq_idx = (i%prepost_depth)*peer_count + complete_idx;
            if (rreq_status[rreq_idx].status == PENDING) {
                ib_progress_recv();
                continue;
            }
            complete_idx++; 
        }

        complete_idx = 0; 
        while (complete_idx < sreq_idx) { 
            if (sreq_status[complete_idx].status == PENDING) {
                ib_progress_send();
                continue;
            } 
            complete_idx++; 
        }
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif

#ifndef _FREE_PACK_
        /*unpack data*/
        boundary_unpack (exchange_buf, unpackbuf[stage_buf_idx], sizex, sizey, threadsperblock, boundary_stream);
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif

#ifndef _FREE_BOUNDARY_COMPUTE_
	/*launch boundary computation*/
        boundary_compute (compute_buf, exchange_buf, sizex, sizey, boundary, threadsperblock, gridsize, boundary_stream);
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
#endif

#ifndef _FREE_SYNC_
	CUDA_CHECK(cudaDeviceSynchronize());
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++); 
#endif

#ifndef _FREE_NETWORK_
        if (i+prepost_depth < iter_count) {
	    int stage_buf_prepost_idx = (i+prepost_depth)%buffering;
	    int rreq_prepost_idx = ((i+prepost_depth)%prepost_depth)*peer_count;
	    packbuf_disp = 0;	

            /*y dim*/
            if (left != MPI_PROC_NULL) { 
                ib_irecv((void *)(unpackbuf[stage_buf_prepost_idx] + packbuf_disp), 
				msg_size_bytes, left, reg_unpack[stage_buf_prepost_idx], 
				&rreq_status[rreq_prepost_idx]);
                rreq_prepost_idx++;
            }
            packbuf_disp += msg_size;
            if (right != MPI_PROC_NULL) { 
                ib_irecv((void *)(unpackbuf[stage_buf_prepost_idx] + packbuf_disp),
                        	msg_size_bytes, right, reg_unpack[stage_buf_prepost_idx], 
                  	    	&rreq_status[rreq_prepost_idx]);
                rreq_prepost_idx++;
            }
            packbuf_disp += msg_size;
            /*x dim*/
            if (bottom != MPI_PROC_NULL) {
                ib_irecv((void *)(unpackbuf[stage_buf_prepost_idx] + packbuf_disp),
                        msg_size_bytes, bottom, reg_unpack[stage_buf_prepost_idx],
                        &rreq_status[rreq_prepost_idx]);
                rreq_prepost_idx++;
            }
            packbuf_disp += msg_size;
            if (top != MPI_PROC_NULL) {
                ib_irecv((void *)(unpackbuf[stage_buf_prepost_idx] + packbuf_disp),
                        msg_size_bytes, top, reg_unpack[stage_buf_prepost_idx],
                        &rreq_status[rreq_prepost_idx]);
                rreq_prepost_idx++;
            }
	}
#endif

#if defined (_ENABLE_DRPROF_)
        PROF(&prof, prof_idx++);
        prof_update(&prof);
        prof_idx = 0;
#endif

	/*intercahnge the compute and communication buffers*/
        temp = exchange_buf; 
	exchange_buf = compute_buf; 
	compute_buf = temp;
    }

    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&time_elapsed, start_event, stop_event));
    if (comm_rank == 0) {
        fprintf(stdout, "%dx%d %8.2lf usec\n", npx, npy, (time_elapsed*1e3 + (time_prepost/1000))/iter_count);

#if defined (_ENABLE_DRPROF_)
        fprintf(stdout, "Prepost depth: %d latency: %8.2lf nsec \n", prepost_depth, time_prepost);
        prof_dump(&prof);
#endif
    }

#ifdef _VALIDATE_
    emulate_on_host (buf_u_h, buf_v_h, sizex, sizey, boundary, 1/*ghost*/,
            comm_rank, left, right, bottom, top, (iter_count + iter_warmup), comm2d);

    cudaMemcpy (buf_u_h, buf_u, buf_size, cudaMemcpyDeviceToHost);
    cudaMemcpy (buf_v_h, buf_v, buf_size, cudaMemcpyDeviceToHost);

    validate (buf_u_h, buf_v_h, sizex, sizey, 1);
#endif

    CUDA_CHECK(cudaFree(buf_u));
    CUDA_CHECK(cudaFree(buf_v));
    for (x=0; x<buffering; x++) { 
        ib_deregister(reg_pack[x]);
        ib_deregister(reg_unpack[x]);

        CUDA_CHECK(cudaFree(packbuf[x]));
        CUDA_CHECK(cudaFree(unpackbuf[x]));
    }

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaStreamDestroy(interior_stream));
    CUDA_CHECK(cudaStreamDestroy(boundary_stream));

    free(sreq_status);
    free(rreq_status);

    return 0;
}

int main (int c, char *v[])
{
    int iter_count, iter_count_large, iter_warmup, iter_warmup_large;
    int dim[2], period[2];
    int reorder, boundary_compute_width;
    int npx, npy, sizex, sizey, count;
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
    if (getenv("COMM_SIZE") != NULL) {
        max_comm_size = atoi(getenv("COMM_SIZE"));
    }
    if (getenv("ITER_COUNT") != NULL) {
        iter_count = atoi(getenv("ITER_COUNT"));
    }
    if (getenv("ITER_COUNT_LARGE") != NULL) {
        iter_count_large = atoi(getenv("ITER_COUNT_LARGE"));
    }
    if (getenv("WARMUP_COUNT") != NULL) {
        iter_warmup = atoi(getenv("WARMUP_COUNT"));
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
    if (getenv("BUFFERING") != NULL) {
        buffering = atoi(getenv("BUFFERING"));
    }
    if (getenv("PREPOST_RECV_DEPTH") != NULL) {
        prepost_depth_default = atoi(getenv("PREPOST_RECV_DEPTH"));
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

    int dev_count = 0, dev_id = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0) {
        fprintf(stderr, "no CUDA devices found \n");
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

    fprintf(stderr, "[%d] using GPU device: %d \n", comm_rank, dev_id);
    cudaSetDevice(dev_id);

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
    //fprintf(stderr, "[%d] left: %d right: %d top: %d bottom: %d \n", comm_rank, left, right, top, bottom);

    if (left != MPI_PROC_NULL) { 
        peers[peer_count] = left;
        peer_count++;
    }
    if (right != MPI_PROC_NULL) { 
        peers[peer_count] = right;
        peer_count++;
    }
    if (top != MPI_PROC_NULL) { 
        peers[peer_count] = top;
        peer_count++;
    }
    if (bottom != MPI_PROC_NULL) { 
        peers[peer_count] = bottom;
        peer_count++;
    }

    MPI_Cart_coords(comm2d, comm_rank, 2, rank_coords);

    MPI_Barrier(MPI_COMM_WORLD);

    /*setup IB communication infrastructure*/
    int ret = SUCCESS;
    ret = setup_ib_domain(comm_rank);
    if (ret != SUCCESS) {
        fprintf(stderr, "setup_ib_domain returned error \n");
        exit(-1);
    }

    /*setup IB connections with peers*/
    ret = setup_ib_connections (comm2d, peers, peer_count); 
    if (ret != SUCCESS) {
        fprintf(stderr, "setup_ib_connections returned error \n");
        exit(-1);
    }

    if (min_size < 4*boundary_compute_width)
         min_size = 4*boundary_compute_width;

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
