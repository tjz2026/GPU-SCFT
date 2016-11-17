#include <cuda_runtime.h>

#include <cufft.h>

#include <helper_cuda.h>
#include <helper_string.h>
#include <helper_functions.h>

#include <vector>
//Header stuff here
#define Pi 3.1415926535

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


#endif //TAUCS_H

