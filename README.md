2D Stencil improved with GPUDirect Async

Description
***********

Simulates computation on a distributed 2D stencil distributed across a 2D grid
of processes. The computation is a simple average where the value of an element
in the current iteration depends on the values of its neighboring elements
(1-cell stencil) from the previous iteration. The near-neighbor exchange
involves transferring the outermost most layer of the data grid between
neighboring processes. There are two data grids, u and v, where one is computed
from the other, alternatingly, in each iteration. This is modeled after the
presence of multiple components in real science problems that affect each other
over time, for example velocity and stress in seismic modeling codes.  The
organization of the main loop is as follows which we undestaood is a typical
way things are done in stencil applications.  

Loop {
     Interior Compute

     Boundary Pack 
  
     Exchange 

     Boundary Unpack 

     Boundary Compute 
}

Interior compute is launched on one CUDA stream while all other activity
happens on another CUDA stream. This allows for overlap of the computation with
the data exchange. There are three versions of the test: 

2dstencil_p2p_sr: Exchange is implemented using MPI (CUDA-aware). The
synchronization between MPI and CUDA is managed by the CPU (for example: makes
sure boundary pack is complete before exchange happens and makes sure
exchange is complete before boundary unpack is called). 

2dstencil_p2p_ib: Exchange is implemented using standard IB verbs. The
synchronization between IB and CUDA is managed by the CPU (for example: makes
sure boundary pack is complete before exchange happens and makes sure
exchange is complete before boundary unpack is called).

2dstencil_p2p_peersync: Exchange is implemented using extended IB verbs with
support for PeerSync.  The synchronization between IB and CUDA is offloaded
onto GPU streams using PeerSync API.

RUN STEPS
*********

The variables to be set in the environment when running the benchmark are: 

PX, PY - the dimensions of the process grid 

PX*PY must match the number of processes in the job

Sample: 

mpirun_rsh -np 4 -hostfile hfile PX=2 PY=2 MV2_USE_CUDA=1 CUDA_VISIBLE_DEVICES=0,1 2dstencil_p2p_sr
