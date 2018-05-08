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

#include <math.h>
#include "common.h"
#include "mpi.h"
#include "validate_strong.h"

#define MAX_ERROR_PRINTS 10

float *vbuf_test = NULL, *ubuf_test = NULL;
int my_rank;
 
void emulate_on_host (float *ubuf, float *vbuf, int sizex, int sizey, int boundary, int ghost, 
	    int comm_rank, int left, int right, int bottom, int top, int iter_count, MPI_Comm comm2d) 
{
    int i, x, y, neighbors; 
    float *ptr, *xp_ptr, *xm_ptr, *yp_ptr, *ym_ptr;
    MPI_Request *sreq, *rreq;  
    int buf_size, msg_elems, msg_size;
    float *packbuf, *unpackbuf; 
    float *compute_buf, *exchange_buf, *temp;
    int packbuf_disp;
    int rreq_idx, sreq_idx; 
	
    neighbors = 4;
 
    my_rank = comm_rank;

    buf_size = sizeof(float)*(sizex + 2)*(sizey + 2);
    msg_elems = (sizex > sizey) ? sizex : sizey;
    msg_size = msg_elems*sizeof(float);

    ubuf_test = malloc (buf_size);
    vbuf_test = malloc (buf_size);
    packbuf = malloc(msg_size*neighbors);
    unpackbuf = malloc(msg_size*neighbors);

    memset(ubuf_test, 0, buf_size);
    memset(vbuf_test, 0, buf_size);

    for (x = 0; x < sizex; x++) { 
        for (y = 0; y < sizey; y++) { 
            *(ubuf_test + (x+1)*(sizey+2) + (y+1)) 
              = *(ubuf + (x+ghost)*(sizey+2*ghost) + (y+ghost));
            *(vbuf_test + (x+1)*(sizey+2) + (y+1)) 
              = *(vbuf + (x+ghost)*(sizey+2*ghost) + (y+ghost));
        }
    }

#if 0
    if (!my_rank) {
        fprintf(stderr, "sizex: %d sizey: %d boundary: %d \n", sizex, sizey, boundary); 
	fprintf(stderr, "printing u ... \n");
        for (x = 0; x < sizex+2; x++) {
            for (y = 0; y < sizey+2; y++) {
                    fprintf(stderr, "%f \t", *(ubuf_test + x*(sizey+2) + y));
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n \n");
 
	fprintf(stderr, "printing v ... \n");
        for (x = 0; x < sizex+2; x++) {
            for (y = 0; y < sizey+2; y++) {
                    fprintf(stderr, "%f \t", *(vbuf_test + x*(sizey+2) + y));
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n \n");
    }
#endif

    /*allocating requests*/
    sreq = (MPI_Request *) malloc(neighbors*sizeof(MPI_Request));
    rreq = (MPI_Request *) malloc(neighbors*sizeof(MPI_Request));

    compute_buf = ubuf_test; 
    exchange_buf = vbuf_test;

    for (i=0; i<iter_count; i++) {
        /*compute interiors*/

	/*exchange ghosts*/

	packbuf_disp = 0;
	rreq_idx = 0;
        /*post all receives*/ 
        /*y dim*/
        if (left != -1) {
            MPI_Irecv((void *)(unpackbuf + packbuf_disp),
                msg_elems, MPI_FLOAT, left, left, comm2d,
                    &rreq[rreq_idx]);
            rreq_idx++;
        }
        packbuf_disp += msg_elems;
        if (right != -1) {
            MPI_Irecv((void *)(unpackbuf + packbuf_disp),
                    msg_elems, MPI_FLOAT, right, right, comm2d,
                    &rreq[rreq_idx]);
            rreq_idx++;
        }
        packbuf_disp += msg_elems;
        /*x dim*/
        if (bottom != -1) {
            MPI_Irecv((void *)(unpackbuf + packbuf_disp),
                msg_elems, MPI_FLOAT, bottom, bottom, comm2d,
                &rreq[rreq_idx]);
            rreq_idx++;
        }
        packbuf_disp += msg_elems;
        if (top != -1) {
            MPI_Irecv((void *)(unpackbuf + packbuf_disp),
                msg_elems, MPI_FLOAT, top, top, comm2d,
                &rreq[rreq_idx]);
            rreq_idx++;
        }
        packbuf_disp += msg_elems;

	/*pack buf*/
	temp = packbuf;
        if (left != -1) { 
           /*left*/
    	   for (x=0; x<sizex; x++) {
               *(temp + x) = *(exchange_buf + (x+1)*(sizey + 2) + 1); 
           }
        }
        temp = temp + msg_elems; 
        if (right != -1) {
           /*right*/
           for (x=0; x<sizex; x++) {
              *(temp + x) = *(exchange_buf + (x+1)*(sizey + 2) + sizey);
           } 
	} 
        temp = temp + msg_elems;
        /*bottom*/
        if (bottom != -1) {
            for (y=0; y<sizey; y++) {
               *(temp + y) = *(exchange_buf + 1*(sizey + 2) + (y+1));
            }
	}
        temp = temp + msg_elems;
        /*top*/
        if (top != -1) {
            for (y=0; y<sizey; y++) {
               *(temp + y) = *(exchange_buf + sizex*(sizey + 2) + (y+1));
            }
	}

#if 0
        int t;
        temp = packbuf;
        fprintf(stderr, "packbuf \n \n");
        fprintf(stderr, "\n \n");
        if (!my_rank)
        for (t = 0; t <msg_elems*4; t++) {
            if (t%msg_elems == 0) fprintf(stderr, "\n \n");
            fprintf(stderr, "%lf \t", temp[t]);
        }
        fprintf(stderr, "\n \n");
        fprintf(stderr, "\n \n");
#endif 


	packbuf_disp = 0;
	sreq_idx = 0;
	/*Post sends*/
        /*y dim*/
        if (left != -1) {
            MPI_Isend((void *)(packbuf + packbuf_disp),
                    msg_elems, MPI_FLOAT, left, comm_rank, comm2d,
                    &sreq[sreq_idx]);
            sreq_idx++;
        }
        packbuf_disp += msg_elems;
        if (right != -1) {
            MPI_Isend((void *)(packbuf + packbuf_disp),
                    msg_elems, MPI_FLOAT, right, comm_rank, comm2d,
                    &sreq[sreq_idx]);
            sreq_idx++;
        }
        packbuf_disp += msg_elems;
        /*x dim*/
        if (bottom != -1) {
            MPI_Isend((void *)(packbuf + packbuf_disp),
                    msg_elems, MPI_FLOAT, bottom, comm_rank, comm2d,
                    &sreq[sreq_idx]);
            sreq_idx++;
        }
        packbuf_disp += msg_elems;
        if (top != -1) {
            MPI_Isend((void *)(packbuf + packbuf_disp),
                msg_elems, MPI_FLOAT, top, comm_rank, comm2d,
                &sreq[sreq_idx]);
            sreq_idx++;
        }

        MPI_Waitall(rreq_idx, rreq, MPI_STATUS_IGNORE);
        MPI_Waitall(sreq_idx, sreq, MPI_STATUS_IGNORE); 

	/*unpack buf*/
        temp = unpackbuf;
        if (left != -1) {
            /*left*/
            for (x=0; x<sizex; x++) {
               *(exchange_buf + (x+1)*(sizey + 2)) = *(temp + x);
            }
	}
        temp = temp + msg_elems;
        if (right != -1) {
            /*right*/
            for (x=0; x<sizex; x++) {
               *(exchange_buf + (x+1)*(sizey + 2) + (sizey+1)) = *(temp + x);
            }
	}
        temp = temp + msg_elems;
        if (bottom != -1) {
            /*bottom*/
            for (y=0; y<sizey; y++) {
               *(exchange_buf + (y+1)) = *(temp + y);
            }
	}
        temp = temp + msg_elems;
        if (top != -1) {
            /*top*/
            for (y=0; y<sizey; y++) {
               *(exchange_buf + (sizex+1)*(sizey + 2) + (y+1)) = *(temp + y);
            }
	}

#if 0
        temp = unpackbuf; 
	fprintf(stderr, "unpackbuf \n \n");
        if (!my_rank) 
	for (t = 0; t <msg_elems*4; t++) { 
            if (t%msg_elems == 0) fprintf(stderr, "\n \n");
	    fprintf(stderr, "%lf \t", temp[t]);
        }
	fprintf(stderr, "\n \n");
	fprintf(stderr, "\n \n");
#endif 

	/*compute interior*/
        for (x=boundary+1; x<sizex-boundary+1; x++) {
            for (y=boundary+1; y<sizey-boundary+1; y++) {
                 ptr = compute_buf + x*(sizey + 2) + y;
                 xp_ptr = exchange_buf + (x + 1)*(sizey + 2) + y;
                 xm_ptr = exchange_buf + (x - 1)*(sizey + 2) + y;
                 yp_ptr = exchange_buf + x*(sizey + 2) + y + 1;
                 ym_ptr = exchange_buf + x*(sizey + 2) + y - 1;

                 *ptr = *ptr + (*xp_ptr + *xm_ptr + *yp_ptr + *ym_ptr)/4.0;
            }
        }

	/*compute boundaries*/     
	/*bottom*/ 
        for (x=1; x<(boundary+1); x++) { 
            for (y=1; y<(sizey+1); y++) { 
                 ptr = compute_buf + x*(sizey + 2) + y; 
                 xp_ptr = exchange_buf + (x + 1)*(sizey + 2) + y; 
                 xm_ptr = exchange_buf + (x - 1)*(sizey + 2) + y; 
                 yp_ptr = exchange_buf + x*(sizey + 2) + y + 1; 
                 ym_ptr = exchange_buf + x*(sizey + 2) + y - 1; 

                 *ptr = *ptr + (*xp_ptr + *xm_ptr + *yp_ptr + *ym_ptr)/4.0;
            } 
        }

	/*top*/
        for (x=(sizex-boundary+1); x<(sizex+1); x++) {
            for (y=1; y<(sizey+1); y++) {
                 ptr = compute_buf + x*(sizey + 2) + y;
                 xp_ptr = exchange_buf + (x + 1)*(sizey + 2) + y;
                 xm_ptr = exchange_buf + (x - 1)*(sizey + 2) + y;
                 yp_ptr = exchange_buf + x*(sizey + 2) + y + 1;
                 ym_ptr = exchange_buf + x*(sizey + 2) + y - 1;

                 *ptr = *ptr + (*xp_ptr + *xm_ptr + *yp_ptr + *ym_ptr)/4.0;
            }
        }

	/*left*/
        for (x=boundary+1; x<sizex-boundary+1; x++) {
            for (y=1; y<(boundary+1); y++) {
                 ptr = compute_buf + x*(sizey + 2) + y;
                 xp_ptr = exchange_buf + (x + 1)*(sizey + 2) + y;
                 xm_ptr = exchange_buf + (x - 1)*(sizey + 2) + y;
                 yp_ptr = exchange_buf + x*(sizey + 2) + y + 1;
                 ym_ptr = exchange_buf + x*(sizey + 2) + y - 1;

                 *ptr = *ptr + (*xp_ptr + *xm_ptr + *yp_ptr + *ym_ptr)/4.0;
            }
        }

	/*right*/
        for (x=boundary+1; x<sizex-boundary+1; x++) {
            for (y=(sizey-boundary+1); y<(sizey+1); y++) {
                 ptr = compute_buf + x*(sizey + 2) + y;
                 xp_ptr = exchange_buf + (x + 1)*(sizey + 2) + y;
                 xm_ptr = exchange_buf + (x - 1)*(sizey + 2) + y;
                 yp_ptr = exchange_buf + x*(sizey + 2) + y + 1;
                 ym_ptr = exchange_buf + x*(sizey + 2) + y - 1;

                 *ptr = *ptr + (*xp_ptr + *xm_ptr + *yp_ptr + *ym_ptr)/4.0;
            }
        }

        temp = exchange_buf;
        exchange_buf = compute_buf;
        compute_buf = temp;
    }

    free(packbuf);
    free(unpackbuf);
    free(sreq);
    free(rreq);
}

void print_actual (float *ubuf, float *vbuf, int sizex, int sizey, int ghost) 
{
    int x, y;

    fprintf(stderr, "in validate, actual stencils \n");
    for (x = 0; x < sizex+2*ghost; x++) {
        for (y = 0; y < sizey+2*ghost; y++) {
              fprintf(stderr, "u[%d][%d]: %f \t", x, y, *(ubuf + x*(sizey+2*ghost) + y));
      }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n \n");

    for (x = 0; x < sizex+2*ghost; x++) {
        for (y = 0; y < sizey+2*ghost; y++) {
              fprintf(stderr, "v[%d][%d]: %f \t", x, y, *(vbuf + x*(sizey+2*ghost) + y));
      }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n \n");
}


void print_all (float *ubuf, float *vbuf, int sizex, int sizey, int ghost) 
{
    int x, y;

    fprintf(stderr, "in validate, actual stencils \n \n");
    fprintf(stderr, "u >>>>>>>>>>>>> \n ");
    for (x = 0; x < sizex+2*ghost; x++) {
        for (y = 0; y < sizey+2*ghost; y++) {
              fprintf(stderr, "%f \t", *(ubuf + x*(sizey+2*ghost) + y));
              //fprintf(stderr, "u[%d][%d]: %f \t", x, y, *(ubuf + x*(sizey+2*ghost) + y));
      }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n \n");

    fprintf(stderr, "v >>>>>>>>>>>>> \n ");

    for (x = 0; x < sizex+2*ghost; x++) {
        for (y = 0; y < sizey+2*ghost; y++) {
              //fprintf(stderr, "v[%d][%d]: %f \t", x, y, *(vbuf + x*(sizey+2*ghost) + y));
              fprintf(stderr, "%f \t", *(vbuf + x*(sizey+2*ghost) + y));
      }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n \n");

    fprintf(stderr, "in validate, expected stencils \n \n");
    fprintf(stderr, "u >>>>>>>>>>>>> \n ");
    for (x = 0; x < sizex+2; x++) {
        for (y = 0; y < sizey+2; y++) {
              //fprintf(stderr, "u[%d][%d]: %f \t", x, y, *(ubuf_test + x*(sizey+2*ghost) + y));
              fprintf(stderr, "%f \t", *(ubuf_test + x*(sizey+2*ghost) + y));
        }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n \n");

    fprintf(stderr, "v >>>>>>>>>>>>> \n ");

    for (x = 0; x < sizex+2; x++) {
        for (y = 0; y < sizey+2; y++) {
              //fprintf(stderr, "v[%d][%d]: %f \t", x, y, *(vbuf_test + x*(sizey+2*ghost) + y));
              fprintf(stderr, "%f \t", *(vbuf_test + x*(sizey+2*ghost) + y));
        }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n \n");
}

void validate (float *ubuf, float *vbuf, int sizex, int sizey, int ghost) 
{
    int x, y;
    float expected, actual;
    int errors = 0;

#if 0
    if (!my_rank)
       print_all (ubuf, vbuf, sizex, sizey, ghost); 
#endif

#if 1 
    if (!my_rank)
    for (x = 0; x < sizex; x++) {
        for (y = 0; y < sizey; y++) {
            expected = *(ubuf_test + (x+1)*(sizey+2) + (y+1));
            actual = *(ubuf + (x+ghost)*(sizey+2*ghost) + (y+ghost));
            if ((fabs(expected - actual) > 0.0001) && (errors <= MAX_ERROR_PRINTS)) {
                 fprintf(stderr, "[%d] validation failed on ubuf for index x: %d y: %d expected: %f actual: %f \n", 
            		my_rank, x, y, expected, actual);
        	 errors++;
            } 
 
            expected = *(vbuf_test + (x+1)*(sizey+2) + (y+1));
            actual = *(vbuf + (x+ghost)*(sizey+2*ghost) + (y+ghost)); 
            if ((fabs(expected - actual) > 0.0001) && (errors <= MAX_ERROR_PRINTS)) {
                 fprintf(stderr, "[%d] validation failed on vbuf index x: %d y: %d expected: %f actual: %f \n",
                            my_rank, x, y, expected, actual);
        	 errors++;
            }
    	}
    }

    if (!my_rank) 
    if (!errors) { 	
        fprintf(stderr, "[%d] validation passed for size: %d %d\n", my_rank, sizex, sizey);	
    } else {
        fprintf(stderr, "[%d] validation failed for size: %d %d\n", my_rank, sizex, sizey);	
    }
#endif

    free(ubuf_test);
    free(vbuf_test);	
}
