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

#define SUCCESS 0
#define FAILURE 1

#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h> 

extern struct prof prof;

enum {
PENDING = 0, 
COMPLETE,
};

/*exchange info*/
typedef struct {
    uint16_t lid;
    uint32_t psn;
    uint32_t qpn;	
} qpinfo_t;

typedef struct {
    uint32_t key; 
    struct ibv_mr *mr;
} ib_reg_t;

/*client resources*/
typedef struct {
   void *region;
   int region_size;
   /*ib related*/
   struct ibv_qp *qp;
   struct ibv_cq *send_cq;
   struct ibv_cq *recv_cq;
   struct ibv_mr *region_mr;
} client_t;

typedef struct {
   uint32_t n_values; 
   volatile uint32_t *ptrs[4];
   uint32_t values[4];
   unsigned flags[4];
} ib_send_info_t;

typedef struct {
   int peer;
   int status;
   ib_send_info_t send_info;
} ib_status_t;

/*request list*/
typedef struct ib_request {
    ib_status_t *status;
    union
    {
        struct ibv_recv_wr rr;
        struct ibv_send_wr sr;
        struct ibv_exp_send_wr sr_exp;
    } in;
    union
    {
        struct ibv_send_wr* bad_sr;
        struct ibv_recv_wr* bad_rr;
        struct ibv_exp_send_wr *bad_sr_exp;
    } out;
    struct ibv_sge sg_entry;
    struct ib_request *next;
} ib_request_t;

/*IB resources*/
typedef struct {
    struct ibv_context *context;
    struct ibv_pd      *pd;
} ib_context_t;

extern client_t *clients;
extern int *client_index;
extern ib_request_t *ib_request_region; 
extern ib_request_t *ib_request_free_list; 
extern int ib_request_active_count; 
extern int ib_request_limit;
extern struct ibv_device *ib_dev;
extern int ib_port;
extern ib_context_t *ib_ctx;
extern int ib_tx_depth;
extern int ib_rx_depth;
extern int ib_inline_size;
extern struct ibv_port_attr ib_port_attr;

ib_request_t *get_ib_request();
void release_ib_request(ib_request_t *req);
ib_reg_t *ib_register(void *addr, size_t length);
int setup_ib_domain (int rank);
int setup_ib_connections (MPI_Comm comm, int *peers, int count);
void ib_irecv (void *buf, int size, int peer, ib_reg_t *ib_reg, ib_status_t *status); 
void ib_isend (void *buf, int size, int peer, ib_reg_t *ib_reg, ib_status_t *status);
void ib_progress (); 
void ib_progress_send (); 
void ib_progress_recv ();
void ib_deregister(ib_reg_t *reg); 
