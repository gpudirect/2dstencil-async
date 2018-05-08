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
#include "pack.h"

__constant__ long long int size_d;
__constant__ long long int sizeinterior_d;
__constant__ long long int size2_d;
__constant__ long long int size2_log_d;
__constant__ long long int size_log_d;
__constant__ long long int sizem1_d;

__constant__ long long int boundary_d;
__constant__ long long int boundary_log_d;

__constant__ long long int sizep2_d;
__constant__ long long int elemsinterior_d;
__constant__ long long int elemsboundary_d;
__constant__ long long int elems2ndboundary_d;

/*Note that the matrix dimensions (Z contiguous) and thread dimensions (X contiguous) are reversed in order*/
__global__ void pack(float *tbuf_x, float *tbuf_y, float *sbuf)
{
    int   sidx, tidx, thidx;

    thidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thidx < size_d) {
	/*pack x*/
        /*pack bottom boundary*/
        sidx = sizep2_d + thidx + 1;
        tidx = thidx;

        tbuf_x[tidx] = sbuf[sidx];

        /*pack top boundary*/
        sidx = size_d*sizep2_d + thidx + 1;
        tidx = size_d + thidx;

        tbuf_x[tidx] = sbuf[sidx];

	/*pack y*/
        /*pack left boundary*/
        sidx = (thidx + 1)*sizep2_d + 1;
        tidx = thidx;

        tbuf_y[tidx] = sbuf[sidx];

        /*pack right boundary*/
        sidx = (thidx + 1)*sizep2_d + size_d;
        tidx = size_d + thidx;

        tbuf_y[tidx] = sbuf[sidx];
   }
}

__global__ void pack_x(float *tbuf, float *sbuf)
{
    int   sidx, tidx, thidx;

    thidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thidx < size_d) { 
        /*pack bottom boundary*/
        sidx = sizep2_d + thidx + 1; 
        tidx = thidx; 
 
        tbuf[tidx] = sbuf[sidx];
          
        /*pack top boundary*/
        sidx = size_d*sizep2_d + thidx + 1; 
        tidx = size_d + thidx; 
       
        tbuf[tidx] = sbuf[sidx];
   }
}

__global__ void pack_y(float *tbuf, float *sbuf)
{
    long long int sidx, tidx, thidx;

    thidx = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (thidx < size_d) {
        /*pack left boundary*/
        sidx = (thidx + 1)*sizep2_d + 1; 
        tidx = thidx; 

        tbuf[tidx] = sbuf[sidx];
          
        /*pack right boundary*/
        sidx = (thidx + 1)*sizep2_d + size_d;
        tidx = size_d + thidx; 
      
        tbuf[tidx] = sbuf[sidx];
    }
}

__global__ void unpack(float *tbuf, float *sbuf_x, float *sbuf_y) {
    int   sidx, tidx, thidx;

    thidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thidx < size_d) {
	/*unpack_x*/
        /*unpack bottom boundary*/
        tidx = thidx + 1;
        sidx = thidx;

        tbuf[tidx] = sbuf_x[sidx];

        /*pack top boundary*/
        tidx = (size_d + 1)*sizep2_d + thidx + 1;
        sidx = size_d + thidx;

        tbuf[tidx] = sbuf_x[sidx];

	/*unpack y*/
        /*unpack left boundary*/
        tidx = (thidx + 1)*sizep2_d;
        sidx = thidx;

        tbuf[tidx] = sbuf_y[sidx];

        /*pack right boundary*/
        tidx = (thidx + 1)*sizep2_d + size_d + 1;
        sidx = size_d + thidx;

        tbuf[tidx] = sbuf_y[sidx];
    }
}

__global__ void unpack_x(float *tbuf, float *sbuf) {
    int   sidx, tidx, thidx;

    thidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thidx < size_d) {
        /*unpack bottom boundary*/
        tidx = thidx + 1;
        sidx = thidx;
 
        tbuf[tidx] = sbuf[sidx];
 
        /*pack top boundary*/
        tidx = (size_d + 1)*sizep2_d + thidx + 1;
        sidx = size_d + thidx;
 
        tbuf[tidx] = sbuf[sidx];
    }
}

__global__ void unpack_y(float *tbuf, float *sbuf) {
    int   sidx, tidx, thidx;

    thidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thidx < size_d) {
        /*unpack left boundary*/
        tidx = (thidx + 1)*sizep2_d;
        sidx = thidx;
 
        tbuf[tidx] = sbuf[sidx];
 
        /*pack right boundary*/
        tidx = (thidx + 1)*sizep2_d + size_d + 1;
        sidx = size_d + thidx;
 
        tbuf[tidx] = sbuf[sidx];
    }
}

__global__ void compute_xboundary(float *tbuf, float *sbuf) {
    int   i, thidx, numthreads, x, y, tmp;
    float *addr, *x_paddr, *x_maddr, *y_paddr, *y_maddr;

    thidx = blockIdx.x * blockDim.x + threadIdx.x;
    numthreads = gridDim.x * blockDim.x;

    for (i = thidx; i < elemsboundary_d; i += numthreads) {
         //y = i >> boundary_log_d;
         //tmp = y << boundary_log_d;
         //x = (i - tmp);
         x = i >> size_log_d;
         tmp = x << size_log_d;
         y = (i - tmp);


         /*note the added 1 to count for the ghost cells*/
         addr = tbuf + (x + 1)*sizep2_d + (y + 1);

         x_paddr = sbuf + (x + 2)*sizep2_d + (y + 1);
         x_maddr = sbuf + (x)*sizep2_d + (y + 1);
         y_paddr = sbuf + (x + 1)*sizep2_d + (y + 2);
         y_maddr = sbuf + (x + 1)*sizep2_d + (y);

         /*sommmmeeee computation*/
         *addr = *addr + ((*x_paddr) + (*x_maddr) +
                    (*y_paddr) + (*y_maddr)) / 4.0;

         /*recalculate x dimension for the other boundary*/
         x = x + (size_d - boundary_d);

         /*note the added 1 to count for the ghost cells*/
         addr = tbuf + (x + 1)*sizep2_d + (y + 1);

         x_paddr = sbuf + (x + 2)*sizep2_d + (y + 1);
         x_maddr = sbuf + (x)*sizep2_d + (y + 1);
         y_paddr = sbuf + (x + 1)*sizep2_d + (y + 2);
         y_maddr = sbuf + (x + 1)*sizep2_d + (y);

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

    for (i = thidx; i < elems2ndboundary_d; i += numthreads) {
         //y = i >> size_log_d;
         //tmp = y << size_log_d;
         //x = (i - tmp);
         x = i >> boundary_log_d;
         tmp = x << boundary_log_d;
         y = (i - tmp);

         /*note the added value to count for the ghost cells*/
         x = x + boundary_d + 1;
	 y = y + 1; 

         addr = tbuf + x*sizep2_d + y;

         x_paddr = sbuf + (x + 1)*sizep2_d + y;
         x_maddr = sbuf + (x - 1)*sizep2_d + y;
         y_paddr = sbuf + x*sizep2_d + (y + 1);
         y_maddr = sbuf + x*sizep2_d + (y - 1);

         /*sommmmeeee computation*/
         *addr = *addr + ((*x_paddr) + (*x_maddr) +
                    (*y_paddr) + (*y_maddr)) / 4.0;

         /*recalculate x dimension for the other boundary*/
         y = y + (size_d - boundary_d);

         addr = tbuf + x*sizep2_d + y;

         x_paddr = sbuf + (x + 1)*sizep2_d + y;
         x_maddr = sbuf + (x - 1)*sizep2_d + y;
         y_paddr = sbuf + x*sizep2_d + (y + 1);
         y_maddr = sbuf + x*sizep2_d + (y - 1);

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
         x = i / sizeinterior_d; 
         tmp = x * sizeinterior_d; 
         y = (i - tmp);

         x = x + boundary_d + 1; 
         y = y + boundary_d + 1; 

         /*note the added value to count for the boundary and ghost cells*/
         addr = tbuf + x*sizep2_d + y;

         x_paddr = sbuf + (x + 1)*sizep2_d + y;
         x_maddr = sbuf + (x - 1)*sizep2_d + y;
         y_paddr = sbuf + x*sizep2_d + (y + 1);
         y_maddr = sbuf + x*sizep2_d + (y - 1);

         /*sommmmeeee computation*/
         *addr = *addr + ((*x_paddr) + (*x_maddr) +
                    (*y_paddr) + (*y_maddr)) / 4.0;
    }
}

extern "C" void boundary_pack (float *tbuf, float * sbuf, long long int len, int threadsperblock, cudaStream_t stream) 
{
    int gridsize;

    if (threadsperblock > len) { 
        threadsperblock = len;
        gridsize = 1;
    } else {
	gridsize = len/threadsperblock + (len&(threadsperblock - 1) > 0);
    }

    pack<<<gridsize, threadsperblock, 0, stream>>>(tbuf + 2*len, tbuf, sbuf);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void boundary_unpack (float *tbuf, float * sbuf, long long int len, int threadsperblock, cudaStream_t stream) 
{
    int gridsize;

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
                 long long int size, long long int boundary, int threadsperblock, int gridsize, cudaStream_t stream) 
{
    int numelems = size*boundary;
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

    numelems = (size-2*boundary)*boundary;
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

extern "C" void interior_compute (float *tbuf, float *sbuf, long long int size, long long int boundary, int threadsperblock, 
			int gridsize, cudaStream_t stream) {
    int numelems = (size - 2*boundary)*(size - 2*boundary); 

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

extern "C" void copytosymbol (long long int size, long long int size_log, long long int size2_log, long long int boundary, long long int boundary_log) {
    long long int size2 = size*size;
    long long int sizep2 = size + 2;
    long long int sizeinterior = size - 2*boundary;
    long long int elemsboundary = size*boundary;
    long long int elemsinterior = sizeinterior * sizeinterior;
    long long int elems2ndboundary = (size-2*boundary)*boundary;

    CUDA_CHECK(cudaMemcpyToSymbol(size_d, &size, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(sizeinterior_d, &sizeinterior, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(sizep2_d, &sizep2, sizeof(long long int), 0, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(size2_d, &size2, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(size_log_d, &size_log, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(size2_log_d, &size2_log, sizeof(long long int), 0, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(boundary_d, &boundary, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(boundary_log_d, &boundary_log, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(elemsinterior_d, &elemsinterior, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(elemsboundary_d, &elemsboundary, sizeof(long long int), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(elems2ndboundary_d, &elems2ndboundary, sizeof(long long int), 0, cudaMemcpyHostToDevice));
}
