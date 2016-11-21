#include <cuda_runtime.h>

#include <cufft.h>

#include <helper_cuda.h>
#include <helper_string.h>
#include <helper_functions.h>

#include <vector>
#include "matrix.h"
//Header stuff here
#define Pi 3.14159265358979

#ifndef TAUCS_H
#define TAUCS_H



// define the data type needed for a GPU device property 
typedef struct {
	int kernal_type;
	
	int GPU_N;
	int *GPU_list;
	cudaStream_t *stream; 
        // cuda stream, Declares a stream handle, a stream is a queue of work to be put on the device, operations with a stream are executed in order.
        // cudaStreamCreate(&stream); will allocate a cuda stream and cudaStreamDestroy(stream); deallocate a stream.
        //Stream is the 4th launch parameter
        // — kernel<<< blocks , threads, smem, stream>>>();
        // Stream is passed into some API calls
        // — cudaMemcpyAsync( dst, src, size, dir, stream);
        // Unless otherwise specified all calls are placed into a default
        // stream
        // — Often referred to as “Stream 0”
        // Stream 0 has special synchronization rules
        // — Synchronous with all streams
        // Operations in stream 0 cannot overlap other streams

	int thread; //!Maximal thread num
	int thread_sur;
	cudaDeviceProp prop[64]; // CUDA device properties;[Data types used by CUDA Runtime] http://docs.nvidia.com/cuda/cuda-runtime-api/
	
}GPU_INFO;


typedef struct{
	//!Grid size for fft transform
	int Nx; 
	int Ny;
	int Nz;
	int Nxh1;
	long NxNyNz;	//!total grid number
	long Nxh1NyNz;
	//! cufft configuration variables
	
	cufftHandle *plan_forward; // forward fft plan 
	cufftHandle *plan_backward;


	//! Temperary variables for cufft, R2C
	std::vector<double*> device_in;	//fft in in each GPU
	std::vector<cufftDoubleComplex*> device_out;// fft out in each GPU;
	
	std::vector<double*> ffl;

}CUFFT_INFO;


typedef struct {
        int Dim; // dimension of grid
	int Nx; // grid point number on the first axis, ( in C vonvention, the slowest axis)
	int Ny;
	int Nz;

} GRID;

typedef struct {
        GRID Cell_grid;
	double Lx; // length of cell on X axis  ( first dimension) 
	double Ly; 
	double Lz;
	double dx;
	double dy;
	double dz;

} CELL;


typedef struct {
     int N_spe; // how many species 
     double XN[3][3]; //the matrix of flory-huggins interaction parameters
     double **W_sp;   // chemical potential of each chemical species
     double **R_sp;   // density of each chemical species
     long Nxyz; // total grid points of real space
     double *exp_w,*exp_w_cu; // exp(-0.5*ds*W(x,y,z))
} CHEMICAL;

typedef struct {
      int N_blk; //how many blocks in a chain
      int Ns;    // discretization of contour variable s over 1
      int *Blk_spe; // block species index, i.e.  identify species index for each block 
      int *Blk_start; // starting index of s for each block
      int *Blk_end;  // ending index of s for each block
      double *Blk_f; //length fraction of each block 
      long Nxyz; // total grid points of real space
      double *qf;  // forward propagator       
      double *qb;
      double Q; // single chain partition function
      double *exp_ksq,*exp_ksq_cu; // exp(-ds*(kx^2+ky^2+kz^2)), on CPU and GPU
} CHAIN;


//extern CHEMICAL AB_melt;
//extern CHAIN diblock;




#endif //TAUCS_H

