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

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "mpi.h"
#include "ib.h"
#include <assert.h>

#define __CU_CHECK(stmt, cond_str)                           \
do {                                                        \
    CUresult result = (stmt);                               \
    if (CUDA_SUCCESS != result) {                           \
        const char *err_str = NULL;                         \
        cuGetErrorString(result, &err_str);                 \
        fprintf(stdout, "[%d:%d] %s", ___FILE__, __LINE__, err_str); \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while (0)

client_t *clients;
int *client_index;
int client_count;
ib_request_t *ib_request_region = NULL;
ib_request_t *ib_request_free_list = NULL;
int ib_request_active_count;
int ib_request_limit = 512;

struct ibv_device *ib_dev = NULL;
int ib_port = 1;
ib_context_t *ib_ctx = NULL;
int ib_tx_depth = 256;
int ib_rx_depth = 256;
int num_cqes = 256;
int ib_inline_size = 64;
struct ibv_port_attr ib_port_attr;

void ib_progress_req (ib_request_t *req) 
{ 
    ib_request_t *temp_req = NULL;

    /*poll until completion*/
    int ne = 0;
    static struct ibv_wc wc;
    do {
        int i;
        for (i=0; i<client_count; i++) { 
            ne = ibv_poll_cq (clients[i].recv_cq, 1, &wc);
            if (ne < 0) {
                fprintf(stderr, "poll_cq returned error \n");
                exit(EXIT_FAILURE);
            } else if (ne) {
                temp_req = (ib_request_t *) wc.wr_id;
                temp_req->status->status = COMPLETE;
                release_ib_request(temp_req);
            }
 
            ne = ibv_poll_cq (clients[i].recv_cq, 1, &wc);
            if (ne < 0) {
                fprintf(stderr, "poll_cq returned error \n");
                exit(EXIT_FAILURE);
            } else if (ne) {
                temp_req = (ib_request_t *) wc.wr_id;
                temp_req->status->status = COMPLETE;
                release_ib_request(temp_req);
            }
	}

        if (temp_req == req) {
            break;
        }
    } while (1);
}

ib_request_t *get_ib_request() 
{
    ib_request_t *req = NULL; 

    if (ib_request_free_list == NULL) { 
        fprintf(stderr, "request_free_list empty!! current limit: %d \n", ib_request_limit);
        return NULL;
    }
   
    req = ib_request_free_list; 
    ib_request_free_list = ib_request_free_list->next; 
    req->next = NULL;

    return req; 
}

void release_ib_request(ib_request_t *req) 
{
    req->next = ib_request_free_list; 

    ib_request_free_list = req;
}


ib_reg_t *ib_register(void *addr, size_t length) {
     ib_reg_t *reg = malloc (sizeof(ib_reg_t)); 

     reg->mr = ibv_reg_mr(ib_ctx->pd, addr, length, 
	    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
            IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

     reg->key = reg->mr->lkey;
     return reg;
}

void ib_deregister(ib_reg_t *reg) {
     ibv_dereg_mr(reg->mr);

     free(reg); 
}

int setup_ib_domain (int rank)
{
    int i, num_devices;
    struct ibv_device **dev_list = NULL;
    const char *select_dev; 
    char *req_dev = NULL;

    if (getenv("USE_IB_HCA") != NULL) {
	req_dev = getenv("USE_IB_HCA");
    }

    dev_list = ibv_get_device_list (&num_devices);
    if (dev_list == NULL) {
        fprintf(stderr, "ibv_get_device_list returned NULL \n");
        return FAILURE;
    }

    ib_dev = dev_list[0];
    if (req_dev != NULL) { 
        for (i=0; i<num_devices; i++) { 
            select_dev = ibv_get_device_name(dev_list[i]);
            if (strstr(select_dev, req_dev) != NULL) { 
                ib_dev = dev_list[i];
		fprintf(stderr, "[%d] using IB device: %s \n", rank, req_dev);
                break;
            }
        }
    }
    if (i == num_devices) {
	select_dev = ibv_get_device_name(dev_list[0]);
        ib_dev = dev_list[0];
	fprintf(stderr, "request device: %s not found, defaulting to %s \n", req_dev, select_dev);  
    }

    /*create context, pd, cq*/
    ib_ctx = malloc (sizeof (ib_context_t));
    if (ib_ctx == NULL) {
        fprintf(stderr, "ib_ctx allocation failed \n");
        return FAILURE;
    }

    ib_ctx->context = ibv_open_device(ib_dev);
    if (ib_ctx->context == NULL) {
        fprintf(stderr, "ibv_open_device failed \n");
        return FAILURE;
    }

    ib_ctx->pd = ibv_alloc_pd (ib_ctx->context);
    if (ib_ctx->pd == NULL) {
        fprintf(stderr ,"ibv_alloc_pd failed \n");
        return FAILURE;
    }

    ibv_query_port (ib_ctx->context, ib_port, &ib_port_attr);

    ib_request_region = malloc (sizeof(ib_request_t)*ib_request_limit);
    ib_request_free_list = ib_request_region;
    for (i=0; i<ib_request_limit-1; i++) {
        ib_request_region[i].next = ib_request_region + i + 1;
    }
    ib_request_region[i].next = NULL;
  
    return SUCCESS;
}

int setup_ib_connections (MPI_Comm comm, int *peers, int count)
{
    struct ibv_qp_init_attr ib_qp_init_attr;
    struct ibv_qp_attr ib_qp_attr;
    int peer, my_index_on_peer;
    int i, flags, comm_size, comm_rank; 
    qpinfo_t *qpinfo_all;
    int ret = SUCCESS; 

    MPI_Comm_size (comm, &comm_size);
    MPI_Comm_rank (comm, &comm_rank);

    if (getenv("IB_CQ_DEPTH") != NULL) {
        num_cqes = atoi(getenv("IB_CQ_DEPTH"));
    }

    client_count = count;

    client_index = malloc(sizeof(int)*comm_size); 
    if (client_index == NULL) { 
        fprintf(stderr, "allocation failed \n");
        return FAILURE;
    }
    memset((void *)client_index, 0, sizeof(int)*comm_size);
  
    clients = malloc(sizeof(client_t)*(client_count+1));
    if (clients == NULL) { 
        fprintf(stderr, "allocation failed \n");
        return FAILURE;
    }

    qpinfo_all = malloc (sizeof(qpinfo_t)*comm_size);
    if (qpinfo_all == NULL) {
        fprintf(stderr, "qpinfo allocation failed \n");
        return FAILURE;
    }

    /*create looback*/
    {
	client_index[comm_rank] = client_count; 

	//create CQs for send/recv
	clients[client_count].send_cq = ibv_create_cq (ib_ctx->context, num_cqes, NULL, NULL, 0);
	if (clients[client_count].send_cq == NULL) {
            fprintf(stderr ,"ibv_create_cq failed \n");
            return FAILURE;
    	}

        clients[client_count].recv_cq = ibv_create_cq (ib_ctx->context, num_cqes, NULL, NULL, 0);
        if (clients[client_count].recv_cq == NULL) {
            fprintf(stderr ,"ibv_create_cq failed \n");
            return FAILURE;
        }
	
        //create QP, set to INIT state and exchange QPN information
        memset(&ib_qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
        ib_qp_init_attr.qp_type = IBV_QPT_RC;
        ib_qp_init_attr.send_cq = clients[client_count].send_cq;
        ib_qp_init_attr.recv_cq = clients[client_count].recv_cq;
        ib_qp_init_attr.cap.max_send_wr  = ib_tx_depth;
        ib_qp_init_attr.cap.max_recv_wr  = ib_rx_depth;
        ib_qp_init_attr.cap.max_send_sge = 1;
        ib_qp_init_attr.cap.max_recv_sge = 1;
        ib_qp_init_attr.cap.max_inline_data = ib_inline_size;
  
        clients[client_count].qp = ibv_create_qp (ib_ctx->pd, &ib_qp_init_attr);
        if (clients[client_count].qp == NULL) { 
            fprintf(stderr, "qp creation failed \n");
            return FAILURE;
        }
  
        memset(&ib_qp_attr, 0, sizeof(struct ibv_qp_attr));
        ib_qp_attr.qp_state        = IBV_QPS_INIT;
        ib_qp_attr.pkey_index      = 0;
        ib_qp_attr.port_num        = ib_port;
        ib_qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | 
         			    IBV_ACCESS_LOCAL_WRITE;
        flags 			  = IBV_QP_STATE | IBV_QP_PKEY_INDEX
         		            | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  
        ret = ibv_modify_qp (clients[client_count].qp, &ib_qp_attr, flags);
        if (ret != 0) {
            fprintf(stderr, "Failed to modify QP to INIT: %d, %s\n", ret, strerror(errno));
            exit(EXIT_FAILURE);
        }
    
  
        ret = ibv_modify_qp (clients[client_count].qp, &ib_qp_attr, flags);
        if (ret != 0) {
            fprintf(stderr, "Failed to modify QP to INIT: %d, %s\n", ret, strerror(errno));
            exit(EXIT_FAILURE);
        }
    
        memset(&ib_qp_attr, 0, sizeof(struct ibv_qp_attr));
        ib_qp_attr.qp_state       = IBV_QPS_RTR;
        ib_qp_attr.path_mtu       = ib_port_attr.active_mtu;
        ib_qp_attr.dest_qp_num    = clients[client_count].qp->qp_num; 
        ib_qp_attr.rq_psn         = 0;
        ib_qp_attr.ah_attr.dlid   = ib_port_attr.lid;
        ib_qp_attr.max_dest_rd_atomic     = 1;
        ib_qp_attr.min_rnr_timer          = 12;
        ib_qp_attr.ah_attr.is_global      = 0;
        ib_qp_attr.ah_attr.sl             = 0;
        ib_qp_attr.ah_attr.src_path_bits  = 0;
        ib_qp_attr.ah_attr.port_num       = ib_port;
        flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU
                 | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN
                 | IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC;
        
        ret = ibv_modify_qp(clients[client_count].qp, &ib_qp_attr, flags);
        if (ret != 0) {
            fprintf(stderr, "Failed to modify RC QP to RTR\n");
            return FAILURE;
        }
    }

    /*creating qps for all peers*/
    for (i=0; i<count; i++) {
        peer = peers[i];

	//create CQs for send/recv
	clients[i].send_cq = ibv_create_cq (ib_ctx->context, num_cqes, NULL, NULL, 0);
	if (clients[i].send_cq == NULL) {
            fprintf(stderr ,"ibv_create_cq failed \n");
            return FAILURE;
    	}

        clients[i].recv_cq = ibv_create_cq (ib_ctx->context, num_cqes, NULL, NULL, 0);
        if (clients[i].recv_cq == NULL) {
            fprintf(stderr ,"ibv_create_cq failed \n");
            return FAILURE;
        }
	
        //create QP, set to INIT state and exchange QPN information
        memset(&ib_qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
        ib_qp_init_attr.qp_type = IBV_QPT_RC;
        ib_qp_init_attr.send_cq = clients[i].send_cq;
        ib_qp_init_attr.recv_cq = clients[i].recv_cq;
        ib_qp_init_attr.cap.max_send_wr  = ib_tx_depth;
        ib_qp_init_attr.cap.max_recv_wr  = ib_rx_depth;
        ib_qp_init_attr.cap.max_send_sge = 1;
        ib_qp_init_attr.cap.max_recv_sge = 1;
        ib_qp_init_attr.cap.max_inline_data = ib_inline_size;
  
        clients[i].qp = ibv_create_qp (ib_ctx->pd, &ib_qp_init_attr);
        if (clients[i].qp == NULL) { 
            fprintf(stderr, "qp creation failed \n");
            return FAILURE;
        }
  
        memset(&ib_qp_attr, 0, sizeof(struct ibv_qp_attr));
        ib_qp_attr.qp_state        = IBV_QPS_INIT;
        ib_qp_attr.pkey_index      = 0;
        ib_qp_attr.port_num        = ib_port;
        ib_qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | 
         			    IBV_ACCESS_LOCAL_WRITE;
        flags 			  = IBV_QP_STATE | IBV_QP_PKEY_INDEX
         		            | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  
        ret = ibv_modify_qp (clients[i].qp, &ib_qp_attr, flags);
        if (ret != 0) {
            fprintf(stderr, "Failed to modify QP to INIT: %d, %s\n", ret, strerror(errno));
            exit(EXIT_FAILURE);
        }
    
  
        ret = ibv_modify_qp (clients[i].qp, &ib_qp_attr, flags);
        if (ret != 0) {
            fprintf(stderr, "Failed to modify QP to INIT: %d, %s\n", ret, strerror(errno));
            exit(EXIT_FAILURE);
        }
    
	qpinfo_all[peer].lid = ib_port_attr.lid;
        qpinfo_all[peer].psn = 0;
        qpinfo_all[peer].qpn = clients[i].qp->qp_num;
    }

    /*exchange qpinfo*/
    MPI_Alltoall(MPI_IN_PLACE, sizeof(qpinfo_t),
		MPI_CHAR, qpinfo_all, sizeof(qpinfo_t), 
		MPI_CHAR, MPI_COMM_WORLD);

    for (i=0; i<count; i++) {
       peer = peers[i];
       client_index[peer] = i;

       memset(&ib_qp_attr, 0, sizeof(struct ibv_qp_attr));
       ib_qp_attr.qp_state       = IBV_QPS_RTR;
       ib_qp_attr.path_mtu       = ib_port_attr.active_mtu;
       ib_qp_attr.dest_qp_num    = qpinfo_all[peer].qpn;
       ib_qp_attr.rq_psn         = qpinfo_all[peer].psn;
       ib_qp_attr.ah_attr.dlid   = qpinfo_all[peer].lid;
       ib_qp_attr.max_dest_rd_atomic     = 1;
       ib_qp_attr.min_rnr_timer          = 12;
       ib_qp_attr.ah_attr.is_global      = 0;
       ib_qp_attr.ah_attr.sl             = 0;
       ib_qp_attr.ah_attr.src_path_bits  = 0;
       ib_qp_attr.ah_attr.port_num       = ib_port;
       flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU
                | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN
                | IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC;

       ret = ibv_modify_qp(clients[i].qp, &ib_qp_attr, flags);
       if (ret != 0) {
           fprintf(stderr, "Failed to modify RC QP to RTR\n");
           return FAILURE;
       }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (i=0; i<count; i++) { 
       peer = peers[i];

       memset(&ib_qp_attr, 0, sizeof(struct ibv_qp_attr));
       ib_qp_attr.qp_state       = IBV_QPS_RTS;
       ib_qp_attr.sq_psn         = 0;
       ib_qp_attr.timeout        = 20;
       ib_qp_attr.retry_cnt      = 7;
       ib_qp_attr.rnr_retry      = 7;
       ib_qp_attr.max_rd_atomic  = 1;
       flags = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT
                | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY
                | IBV_QP_MAX_QP_RD_ATOMIC;

       ret = ibv_modify_qp(clients[i].qp, &ib_qp_attr, flags);
       if (ret != 0)
       {
           fprintf(stderr, "Failed to modify RC QP to RTS\n");
           return FAILURE;
       }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    free(qpinfo_all);

    return SUCCESS;
}

void ib_irecv (void *buf, int size, int peer, ib_reg_t *ib_reg, ib_status_t *status) 
{
    ib_request_t *req = get_ib_request();
    int ret = 0; 

    status->status = PENDING;
    status->peer = peer;

    req->status = status; 

    req->in.rr.next = NULL;
    req->in.rr.wr_id = (uintptr_t) req;
    req->in.rr.num_sge = 1;
    req->in.rr.sg_list = &(req->sg_entry);

    req->sg_entry.length = size;
    req->sg_entry.lkey = ib_reg->key;
    req->sg_entry.addr = (uintptr_t)(buf);

    client_t *client = &clients[client_index[peer]];
    ret = ibv_post_recv (client->qp, &req->in.rr, 
			 &req->out.bad_rr);
    if (ret) { 
        fprintf(stderr, "posting recv failed ret: %d error: %s peer: %d index: %d \n", ret, strerror(errno), peer, client_index[peer]);
	exit(EXIT_FAILURE);
    }
}

void ib_isend (void *buf, int size, int peer, ib_reg_t *ib_reg, ib_status_t *status) 
{
    ib_request_t *req = get_ib_request();
    int ret = 0; 

    status->status = PENDING;
    status->peer = peer;

    req->status = status; 

    req->in.sr.next = NULL;
    req->in.sr.send_flags = IBV_SEND_SIGNALED;
    req->in.sr.opcode = IBV_WR_SEND;
    req->in.sr.wr_id = (uintptr_t) req;
    req->in.sr.num_sge = 1;
    req->in.sr.sg_list = &(req->sg_entry);

    req->sg_entry.length = size;
    req->sg_entry.lkey = ib_reg->key;
    req->sg_entry.addr = (uintptr_t)(buf);

    client_t *client = &clients[client_index[peer]];



    ret = ibv_post_send (client->qp, &req->in.sr, 
			 &req->out.bad_sr);
    if (ret) { 
        fprintf(stderr, "posting send failed: %s \n", strerror(errno));
	exit(EXIT_FAILURE);
    }
}

void ib_progress () 
{ 
    int ne;
    static struct ibv_wc wc;
    ib_request_t *req; 
    int i; 

    for (i=0; i<client_count; i++) { 
        ne = ibv_poll_cq (clients[i].recv_cq, 1, &wc);
        if (ne < 0) {
            fprintf(stderr, "poll_cq returned error \n");
            exit(EXIT_FAILURE);
        } else if (ne) {
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "recv error to wc.status: %d wc_opcode=%d\n", wc.status, wc.opcode );
		exit(-1);
	    }
	    assert(wc.status == IBV_WC_SUCCESS);
            req = (ib_request_t *) wc.wr_id; 
            req->status->status = COMPLETE;
            release_ib_request(req);
        }

        ne = ibv_poll_cq (clients[i].send_cq, 1, &wc);
        if (ne < 0) {
            fprintf(stderr, "poll_cq returned error \n");
            exit(EXIT_FAILURE);
        } else if (ne) {
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "send error to wc.status: %d wc_opcode=%d\n", wc.status, wc.opcode );
		exit(-1);
            }
	    assert(wc.status == IBV_WC_SUCCESS);
            req = (ib_request_t *) wc.wr_id; 
            req->status->status = COMPLETE;
            release_ib_request(req);
        }
    }
}

void ib_progress_send () 
{ 
    int ne;
    static struct ibv_wc wc;
    ib_request_t *req; 
    int i; 

    for (i=0; i<client_count; i++) { 
        ne = ibv_poll_cq (clients[i].send_cq, 1, &wc);
        if (ne < 0) {
            fprintf(stderr, "poll_cq returned error \n");
            exit(EXIT_FAILURE);
        } else if (ne) {
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "send error to wc.status: %d wc_opcode=%d\n", wc.status, wc.opcode );
		exit(-1);
            }
	    assert(wc.status == IBV_WC_SUCCESS);
            req = (ib_request_t *) wc.wr_id; 
            req->status->status = COMPLETE;
            release_ib_request(req);
        }
    }
}

void ib_progress_recv () 
{ 
    int ne;
    static struct ibv_wc wc;
    ib_request_t *req; 
    int i; 

    for (i=0; i<client_count; i++) { 
        ne = ibv_poll_cq (clients[i].recv_cq, 1, &wc);
        if (ne < 0) {
            fprintf(stderr, "poll_cq returned error \n");
            exit(EXIT_FAILURE);
        } else if (ne) {
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "recv error to wc.status: %d wc_opcode=%d\n", wc.status, wc.opcode );
		exit(-1);
	    }
	    assert(wc.status == IBV_WC_SUCCESS);
            req = (ib_request_t *) wc.wr_id; 
            req->status->status = COMPLETE;
            release_ib_request(req);
        }
    }
}
