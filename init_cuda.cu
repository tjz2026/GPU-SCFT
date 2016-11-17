
#include"struct.h"

#include "init_cuda.h"

#include "cuda_aid.cuh"
#include <complex.h>

extern void init_cuda(GPU_INFO *gpu_info,int display){
	
	int gpu_count;
	//cudaDeviceProp prop[64];
	int *gpuid;
	int i;
	gpu_count=0;
	
	gpuid=(int*)malloc(sizeof(int));

	
	//checkCudaErrors(cudaGetDeviceCount(&gpu_info->GPU_N));

	//if(gpu_info->GPU_N==8) gpu_info->GPU_N=4;//! Set the number of GPU

		
	
	gpu_info->GPU_list=(int*)malloc(sizeof(int)*(gpu_info->GPU_N));

	for(i=0;i<(gpu_info->GPU_N);i++) {
		gpu_info->GPU_list[i]=i;}	//!Define on these GPU to calculate 

	
	for (i=0; i < gpu_info->GPU_N; i++){
		
        	checkCudaErrors(cudaGetDeviceProperties(&gpu_info->prop[i], i)); // get the device properties for the specified device number i
		
		checkCudaErrors(cudaSetDevice(gpu_info->GPU_list[i])); // cuda runtime API, thread-safe, to select which GPU to execute CUDA calls on
		
		// Only boards based on Fermi can support P2P
		
            	gpuid[gpu_count++] = gpu_info->GPU_list[i];
                printf("Device Number: %d\n", i);
                printf("  Device name: %s\n", gpu_info->prop[i].name);
                printf("  Memory Clock Rate (KHz): %d\n",\
                       gpu_info->prop[i].memoryClockRate);
                printf("  Memory Bus Width (bits): %d\n",\
                       gpu_info->prop[i].memoryBusWidth);
                printf("  Peak Memory Bandwidth (GB/s): %f\n\n",\
                       2.0*gpu_info->prop[i].memoryClockRate*(gpu_info->prop[i].memoryBusWidth/8)/1.0e6);

                printf("Total global memory:           %lu\n",  gpu_info->prop[i].totalGlobalMem);
                printf("Total shared memory per block: %lu\n",  gpu_info->prop[i].sharedMemPerBlock);
                printf("Total registers per block:     %d\n",  gpu_info->prop[i].regsPerBlock);
                printf("Warp size:                     %d\n",  gpu_info->prop[i].warpSize);
                printf("Maximum memory pitch:          %lu\n",  gpu_info->prop[i].memPitch);
                printf("Maximum threads per block:     %d\n",  gpu_info->prop[i].maxThreadsPerBlock);
                for (int j = 0; j < 3; ++j)
                    printf("Maximum dimension %d of block:  %d\n", j, gpu_info->prop[i].maxThreadsDim[j]);
                for (int j = 0; j < 3; ++j)
                    printf("Maximum dimension %d of grid:   %d\n", j, gpu_info->prop[i].maxGridSize[j]);



		
		//gpu_info->thread=gpu_info->prop[i].maxThreadsPerBlock;
		
}

}


extern void initialize_cufft(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info){

	
	int Dim[3];
	int i;
	int rank = 3;
	int Nx=cufft_info->Nx;
	int Ny=cufft_info->Ny;
	int Nz=cufft_info->Nz;
	
	long NxNyNz=Nx*Ny*Nz;  //,ijk;
	
	cufft_info->NxNyNz=NxNyNz;
	cufft_info->Nxh1=Nx/2+1; // R2C, first dimension is cut in half for reduandency
	cufft_info->Nxh1NyNz=cufft_info->Nxh1*Ny*Nz; // only near half size of grid in complex fft space
	int batch=1;
	
        // doing the factor decompose to determine the thread grid dimension?
	//printf("gpu_info->thread_sur %d\n",gpu_info->thread_sur);
	
	//char comment[200];

	
	//!----------- Initialize GPU memery settings. ------------------------------------------------------	
	
	//int nGPUs = gpu_info->GPU_N;
	
	cufft_info->device_in.resize(gpu_info->GPU_N);
	cufft_info->device_out.resize(gpu_info->GPU_N);
	//-----------! Initialize CUFFT settings. ------------------------------------------------------
	
	dim3 grid(cufft_info->Nx,cufft_info->Ny,cufft_info->Nz),block(1,1,1); // set the grid dimension as (Nx,Ny,Nz). one thread per block
	
	Dim[0]=Nz;Dim[1]=Ny;Dim[2]=Nx;

	cufft_info->plan_forward=(cufftHandle *)malloc(sizeof(cufftHandle)*gpu_info->GPU_N);
	cufft_info->plan_backward=(cufftHandle *)malloc(sizeof(cufftHandle)*gpu_info->GPU_N);

	for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){	
		
		checkCudaErrors(cudaSetDevice(gpu_info->GPU_list[gpu_index]));
	
		checkCudaErrors(cufftCreate(&cufft_info->plan_forward[gpu_index]));
		checkCudaErrors(cufftCreate(&cufft_info->plan_backward[gpu_index]));
		
		
		if(rank==3){
			
			checkCudaErrors(cufftPlanMany (&cufft_info->plan_forward[gpu_index], rank, Dim, NULL, 1, 1, NULL, 1, 1, CUFFT_D2Z, batch));
			checkCudaErrors(cufftPlanMany (&cufft_info->plan_backward[gpu_index], rank, Dim, NULL, 1, 1, NULL, 1, 1, CUFFT_Z2D, batch));
		
		}
		else if(rank==2) {
		
			checkCudaErrors(cufftPlanMany (&cufft_info->plan_forward[gpu_index], rank, Dim, NULL, 1, 1, NULL, 1, 1, CUFFT_D2Z, batch));
			checkCudaErrors(cufftPlanMany (&cufft_info->plan_backward[gpu_index], rank, Dim, NULL, 1, 1, NULL, 1, 1, CUFFT_Z2D, batch));

		}
	}
	
	
	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed [  ]");
	printf("Wonderful We have successfully initialized cufft setting.\n");

	//-----------! Initialize malloc and initilize on CPU. ------------------------------------------------------	
	gpu_info->stream=(cudaStream_t*)malloc( sizeof(cudaStream_t)*gpu_info->GPU_N);
	
	
	printf("Wonderful We have successfully initialized CPU setting.\n");
	
	//-----------! Initialize malloc and initilize on each GPUs. ------------------------------------------------------	

	for (i=0; i < gpu_info->GPU_N; i++){

		checkCudaErrors(cudaSetDevice(gpu_info->GPU_list[i]));
		checkCudaErrors(cudaStreamCreate(&gpu_info->stream[i]));
		checkCudaErrors(cufftSetStream(cufft_info->plan_forward[i], gpu_info->stream[i]));
		checkCudaErrors(cufftSetStream(cufft_info->plan_backward[i], gpu_info->stream[i]));
		
	//	checkCudaErrors(cudaMallocManaged((void**)&(cufft_info->kxyzdz_cu[i]), sizeof(double)* NxNyNz));
		checkCudaErrors(cudaMalloc(&(cufft_info->device_in[i]), sizeof(double)* cufft_info->NxNyNz*batch));
		checkCudaErrors(cudaMalloc(&(cufft_info->device_out[i]), sizeof(cufftDoubleComplex)* cufft_info->Nxh1NyNz*batch));

		checkCudaErrors(cudaDeviceSynchronize());
		
		
	}
	
	
	
	printf("Wonderful We have successfully initialized all the data.\n");
	
	
}


extern void test_cufft(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info){
       double *h_in;
       double complex *h_out;
       cufftDoubleReal *d_in;
       cufftDoubleComplex *d_out;
       int Nx=cufft_info->Nx;
       int Ny=cufft_info->Ny;
       int Nz=cufft_info->Nz;
       h_in = (double*) malloc(sizeof(double) * Nx*Ny*Nz);
       h_out = (double complex*) malloc(sizeof(double complex) * Nx*Ny*(Nz/2+1));   
       unsigned int in_mem_size = Nx*Ny*Nz*sizeof(cufftDoubleReal);
       unsigned int out_mem_size = Nx*Ny*(Nz/2 + 1)*sizeof(cufftDoubleComplex);
       checkCudaErrors(cudaMalloc((void **)&d_in, in_mem_size));
       checkCudaErrors(cudaMalloc((void **)&d_out, out_mem_size));
       int i,j,k,ijk;
       for (i=0, ijk=0; i < Nx; i++){
          for (j=0; j < Ny; j++){
             for (k=0; k < Nz; k++){
                 h_in[ijk]=(i+j+k)*1.0;
                 ijk++;
                                   }
                                 }
                              }

       checkCudaErrors(cudaMemcpy(d_in, h_in, in_mem_size, cudaMemcpyHostToDevice));
       if (cufftExecD2Z(cufft_info->plan_forward[0], d_in, d_out) != CUFFT_SUCCESS){ 
         fprintf(stderr, "CUFFT Error: Unable to execute plan\n"); 
                                                 return;   } 
       if (cufftExecZ2D(cufft_info->plan_backward[0], d_out, d_in) != CUFFT_SUCCESS){ 
         fprintf(stderr, "CUFFT Error: Unable to execute plan\n"); 
                                                 return;   } 
       if (cudaDeviceSynchronize() != cudaSuccess){ 
       fprintf(stderr, "Cuda error: Failed to synchronize\n"); 
                                          return; }
       checkCudaErrors(cudaMemcpy(h_out, d_out, out_mem_size, cudaMemcpyDeviceToHost));
       checkCudaErrors(cudaMemcpy(h_in, d_in, in_mem_size, cudaMemcpyDeviceToHost));
       printf("h_out[0,0,0]:  = %.2f %+.2fi\n", creal(h_out[0]), cimag(h_out[0]));
       printf("h_in[0,0,1]:  = %.2f\n", h_in[1]/(Nx*Ny*Nz));


}


extern void finalize_cufft(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info){
	int i;  //
	//int can_access_peer_0_1;
	
	//! free memery on GPU
	

	
	for (i=0; i < gpu_info->GPU_N; i++){
		
		checkCudaErrors(cudaSetDevice(gpu_info->GPU_list[i]));

		checkCudaErrors(cufftDestroy(cufft_info->plan_forward[i]));
		checkCudaErrors(cufftDestroy(cufft_info->plan_backward[i]));
		
		checkCudaErrors(cudaFree(cufft_info->device_in[i]));
		checkCudaErrors(cudaFree(cufft_info->device_out[i]));
	
		cudaDeviceSynchronize();
	
	}
	
	
	
	//! free memery on CPU
	
	free(gpu_info->stream);
	free(gpu_info->GPU_list);

	printf("Wonderful We have successfully evaculate all the memery on GPU and CPU \n");
	cudaDeviceReset();
}



