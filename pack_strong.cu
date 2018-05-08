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

#include "common.h"
#include "pack_strong.h"

__constant__ long long int size_d;
__constant__ long long int sizex_d;
__constant__ long long int sizey_d;
__constant__ long long int sizexinterior_d;
__constant__ long long int sizeyinterior_d;
__constant__ long long int sizexy_d;
__constant__ long long int sizexy_log_d;
__constant__ long long int sizex_log_d;
__constant__ long long int sizey_log_d;
__constant__ long long int sizexm1_d;
__constant__ long long int sizeym1_d;

__constant__ long long int boundary_d;
__constant__ long long int boundary_log_d;

__constant__ long long int sizexp2_d;
__constant__ long long int sizeyp2_d;
__constant__ long long int elemsinterior_d;
__constant__ long long int elemsxboundary_d;
__constant__ long long int elemsyboundary_d;

/*Note that the matrix dimensions (Z contiguous) and thread dimensions (X contiguous) are reversed in order*/
__global__ void pack(float *tbuf_x, float *tbuf_y, float *sbuf)
{
    int   sidx, tidx, thidx;

    thidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thidx < sizex_d) {
	/*pack y*/
        /*pack left boundary*/
        sidx = (thidx + 1)*sizeyp2_d + 1;
        tidx = thidx;

        tbuf_y[tidx] = sbuf[sidx];

        /*pack right boundary*/
        sidx = (thidx + 1)*sizeyp2_d + sizey_d;
        tidx = size_d + thidx;

        tbuf_y[tidx] = sbuf[sidx];
   }
    if (thidx < sizey_d) {
	/*pack x*/
        /*pack bottom boundary*/
        sidx = sizeyp2_d + thidx + 1;
        tidx = thidx;

        tbuf_x[tidx] = sbuf[sidx];

        /*pack top boundary*/
        sidx = sizex_d*sizeyp2_d + thidx + 1;
        tidx = size_d + thidx;

        tbuf_x[tidx] = sbuf[sidx];
    }
}

__global__ void unpack(float *tbuf, float *sbuf_x, float *sbuf_y) {
    int   sidx, tidx, thidx;

    thidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thidx < sizex_d) {
        /*unpack y*/
        /*unpack left boundary*/
        tidx = (thidx + 1)*sizeyp2_d;
        sidx = thidx;

        tbuf[tidx] = sbuf_y[sidx];

        /*pack right boundary*/
        tidx = (thidx + 1)*sizeyp2_d + sizey_d + 1;
        sidx = size_d + thidx;

        tbuf[tidx] = sbuf_y[sidx];
    }
    if (thidx < sizey_d) {
	/*unpack_x*/
        /*unpack bottom boundary*/
        tidx = thidx + 1;
        sidx = thidx;

        tbuf[tidx] = sbuf_x[sidx];

        /*pack top boundary*/
        tidx = (sizex_d + 1)*sizeyp2_d + thidx + 1;
        sidx = size_d + thidx;

        tbuf[tidx] = sbuf_x[sidx];
    }
}

__global__ void compute_xboundary(float *tbuf, float *sbuf) {
    int   i, thidx, numthreads, x, y, tmp;
    float *addr, *x_paddr, *x_maddr, *y_paddr, *y_maddr;

    thidx = blockIdx.x * blockDim.x + threadIdx.x;
    numthreads = gridDim.x * blockDim.x;

    for (i = thidx; i < elemsxboundary_d; i += numthreads) {
         x = i >> sizey_log_d;
         tmp = x << sizey_log_d;
         y = (i - tmp);


         /*note the added 1 to count for the ghost cells*/
         addr = tbuf + (x + 1)*sizeyp2_d + (y + 1);

         x_paddr = sbuf + (x + 2)*sizeyp2_d + (y + 1);
         x_maddr = sbuf + (x)*sizeyp2_d + (y + 1);
         y_paddr = sbuf + (x + 1)*sizeyp2_d + (y + 2);
         y_maddr = sbuf + (x + 1)*sizeyp2_d + (y);

         /*sommmmeeee computation*/
         *addr = *addr + ((*x_paddr) + (*x_maddr) +
                    (*y_paddr) + (*y_maddr)) / 4.0;

         /*recalculate x dimension for the other boundary*/
         x = x + (sizex_d - boundary_d);

         /*note the added 1 to count for the ghost cells*/
         addr = tbuf + (x + 1)*sizeyp2_d + (y + 1);

         x_paddr = sbuf + (x + 2)*sizeyp2_d + (y + 1);
         x_maddr = sbuf + (x)*sizeyp2_d + (y + 1);
         y_paddr = sbuf + (x + 1)*sizeyp2_d + (y + 2);
         y_maddr = sbuf + (x + 1)*sizeyp2_d + (y);

         /*sommmmeeee computation*/
         *addr = *addr + ((*x_paddr) + (*x_maddr) +
                    (*y_paddr) + (*y_maddr)) / 4.0;
    }
}

__global__ void compute_yboundary(float *tbuf, float *sbuf) {
    int   i, thidx, numthreads, x, y, tmp;
    float *addr, *x_paddr, *x_maddr, *y_paddr, *y_maddr;

    thidx = blockIdx.x * blockDim.x + threadIdx.x;
    numthreads = gridDim.x * blockDim.x;

    for (i = thidx; i < elemsyboundary_d; i += numthreads) {
         x = i >> boundary_log_d;
         tmp = x << boundary_log_d;
         y = (i - tmp);

         /*note the added value to count for the ghost cells*/
         x = x + boundary_d + 1;
	 y = y + 1; 

         addr = tbuf + x*sizeyp2_d + y;

         x_paddr = sbuf + (x + 1)*sizeyp2_d + y;
         x_maddr = sbuf + (x - 1)*sizeyp2_d + y;
         y_paddr = sbuf + x*sizeyp2_d + (y + 1);
         y_maddr = sbuf + x*sizeyp2_d + (y - 1);

         /*sommmmeeee computation*/
         *addr = *addr + ((*x_paddr) + (*x_maddr) +
                    (*y_paddr) + (*y_maddr)) / 4.0;

         /*recalculate x dimension for the other boundary*/
         y = y + (sizey_d - boundary_d);

         addr = tbuf + x*sizeyp2_d + y;

         x_paddr = sbuf + (x + 1)*sizeyp2_d + y;
         x_maddr = sbuf + (x - 1)*sizeyp2_d + y;
         y_paddr = sbuf + x*sizeyp2_d + (y + 1);
         y_maddr = sbuf + x*sizeyp2_d + (y - 1);

         /*sommmmeeee computation*/
         *addr = *addr + ((*x_paddr) + (*x_maddr) +
                    (*y_paddr) + (*y_maddr)) / 4.0;
    }
}

__global__ void compute_interior (float *tbuf, float *sbuf) {
    int   i, thidx, numthreads, x, y, tmp;
    float *addr, *x_paddr, *x_maddr, *y_paddr, *y_maddr;

    thidx = blockIdx.x * blockDim.x + threadIdx.x;
    numthreads = gridDim.x * blockDim.x;

    for (i = thidx; i < elemsinterior_d; i += numthreads) {
         x = i / sizeyinterior_d; 
         tmp = x * sizeyinterior_d; 
         y = (i - tmp);

         x = x + boundary_d + 1; 
         y = y + boundary_d + 1; 

         /*note the added value to count for the boundary and ghost cells*/
         addr = tbuf + x*sizeyp2_d + y;

         x_paddr = sbuf + (x + 1)*sizeyp2_d + y;
         x_maddr = sbuf + (x - 1)*sizeyp2_d + y;
         y_paddr = sbuf + x*sizeyp2_d + (y + 1);
         y_maddr = sbuf + x*sizeyp2_d + (y - 1);

         /*sommmmeeee computation*/
         *addr = *addr + ((*x_paddr) + (*x_maddr) +
                    (*y_paddr) + (*y_maddr)) / 4.0;
    }
}

extern "C" void boundary_pack (float *tbuf, float * sbuf, long long int lenx, long long int leny, int threadsperblock, cudaStream_t stream) 
{
    int gridsize;
    int len;

    len = lenx > leny ? lenx : leny; 

    if (threadsperblock > len) { 
        threadsperblock = len;
        gridsize = 1;
    } else {
	gridsize = len/threadsperblock + (len&(threadsperblock - 1) > 0);
    }

    pack<<<gridsize, threadsperblock, 0, stream>>>(tbuf + 2*len, tbuf, sbuf);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void boundary_unpack (float *tbuf, float * sbuf, long long int lenx, long long int leny, int threadsperblock, cudaStream_t stream) 
{
    int gridsize;
    int len;

    len = lenx > leny ? lenx : leny;

    if (threadsperblock > len) {
        threadsperblock = len;
        gridsize = 1;
    } else {
        gridsize = len/threadsperblock + (len&(threadsperblock - 1) > 0);
    }

    unpack<<<gridsize, threadsperblock, 0, stream>>>(tbuf, sbuf + 2*len, sbuf);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void boundary_compute (float *tbuf, float * sbuf,
                 long long int sizex, long long int sizey, long long int boundary, int threadsperblock, int gridsize, cudaStream_t stream) 
{
    int numelems = sizey*boundary;

    if (threadsperblock > numelems) { 
        threadsperblock = numelems;
        gridsize = 1;
    } else {
        if (gridsize > (numelems)/threadsperblock) {
	    gridsize = (numelems)/threadsperblock + (numelems&(threadsperblock - 1) > 0);	
	}
    }
  
    /*top and bottom*/
    compute_xboundary<<<gridsize, threadsperblock, 0, stream>>>(tbuf, sbuf);
    CUDA_CHECK(cudaGetLastError());

    numelems = (sizex-2*boundary)*boundary;
    if (threadsperblock > numelems) { 
        threadsperblock = numelems;
        gridsize = 1;
    } else {
        if (gridsize > (numelems)/threadsperblock) {
	    gridsize = (numelems)/threadsperblock + (numelems&(threadsperblock - 1) > 0);	
	}
    }

    /*left and right*/
    compute_yboundary<<<gridsize, threadsperblock, 0, stream>>>(tbuf, sbuf);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void interior_compute (float *tbuf, float *sbuf, long long int sizex, long long int sizey, long long int boundary, int threadsperblock, 
			int gridsize, cudaStream_t stream) {
    int numelems = (sizex - 2*boundary)*(sizey - 2*boundary); 

    if (threadsperblock > numelems) { 
        threadsperblock = numelems;
        gridsize = 1;
    } else {
        if (gridsize > (numelems)/threadsperblock) {
            gridsize = (numelems)/threadsperblock + (numelems&(threadsperblock - 1) > 0);
        }
    }

    compute_interior<<<gridsize, threadsperblock, 0, stream>>>(tbuf, sbuf);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void copytosymbol (long long int sizex, long long int sizey, long long int boundary, long long int sizex_log, 
			long long int sizey_log, long long int boundary_log, long long int sizexy_log) {
    long long int sizexy = sizex*sizey;
    long long int sizexp2 = sizex + 2;
    long long int sizeyp2 = sizey + 2;
    long long int sizexinterior = sizex - 2*boundary;
    long long int sizeyinterior = sizey - 2*boundary;
    long long int elemsinterior = sizexinterior * sizeyinterior;
    long long int elemsxboundary = sizey*boundary;
    long long int elemsyboundary = (sizex-2*boundary)*boundary;
    long long int size = (sizex > sizey) ? sizex : sizey; 
 
    CUDA_CHECK(cudaMemcpyToSymbol(size_d, &size, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(sizex_d, &sizex, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(sizey_d, &sizey, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(sizexinterior_d, &sizexinterior, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(sizeyinterior_d, &sizeyinterior, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(sizexp2_d, &sizexp2, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(sizeyp2_d, &sizeyp2, sizeof(long long int), 0, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(sizexy_d, &sizexy, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(sizex_log_d, &sizex_log, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(sizey_log_d, &sizey_log, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(sizexy_log_d, &sizexy_log, sizeof(long long int), 0, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(boundary_d, &boundary, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(boundary_log_d, &boundary_log, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(elemsinterior_d, &elemsinterior, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(elemsxboundary_d, &elemsxboundary, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(elemsyboundary_d, &elemsyboundary, sizeof(long long int), 0, cudaMemcpyHostToDevice));
}
