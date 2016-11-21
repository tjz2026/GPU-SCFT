
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

	int dev_indx[gpu_info->GPU_N];
        // user defined GPU device index, check which GPU to use by type "nvidia-smi"
        assert(gpu_info->GPU_N==1);
        dev_indx[0]=1;

	for (i=0; i < gpu_info->GPU_N; i++){
		
        	checkCudaErrors(cudaGetDeviceProperties(&gpu_info->prop[i], dev_indx[i])); // get the device properties for the specified device number i
		
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


extern void init_chain_chemical(GPU_INFO *gpu_info,GRID *grid,CELL *cell,CHEMICAL *chemical,CHAIN *chain) {
        double *kx,*ky,*kz;  
        double dx,dy,dz,ksq;
        int Nx,Ny,Nz,i,j,k;
        long NxNyNz,ijk;
        Nx=grid->Nx;   
        Ny=grid->Ny;   
        Nz=grid->Nz;   
        NxNyNz=Nx*Nx*Nz;   
	kx=(double *)malloc(sizeof(double)*Nx);
	ky=(double *)malloc(sizeof(double)*Ny);
	kz=(double *)malloc(sizeof(double)*Nz);
	dx=cell->dx;
	dy=cell->dy;
	dz=cell->dz;
	
	chain->exp_ksq=(double *)malloc(sizeof(double)*NxNyNz);	
	chemical->exp_w=(double *)malloc(sizeof(double)*NxNyNz*chemical->N_spe);	
	for(i=0;i<=Nx/2-1;i++) kx[i]=2*Pi*i*1.0/Nx/dx;
	for(i=Nx/2;i<Nx;i++)   kx[i]=2*Pi*(i-Nx)*1.0/dx/Nx;
	for(i=0;i<Nx;i++)      kx[i]*=kx[i];

	for(i=0;i<=Ny/2-1;i++) ky[i]=2*Pi*i*1.0/Ny/dy;
	for(i=Ny/2;i<Ny;i++)   ky[i]=2*Pi*(i-Ny)*1.0/dy/Ny;
	for(i=0;i<Ny;i++)      ky[i]*=ky[i];

	for(i=0;i<=Nz/2-1;i++) kz[i]=2*Pi*i*1.0/Nz/dz;
	for(i=Nz/2;i<Nz;i++)   kz[i]=2*Pi*(i-Nz)*1.0/dz/Nz;
	for(i=0;i<Nz;i++)      kz[i]*=kz[i];
	double ds;
        ds=1.0/chain->Ns;
	for(k=0;k<Nz;k++) {
	   for(j=0;j<Ny;j++){
	      for(i=0;i<Nx;i++){
		ijk=(long)((k*Ny+j)*Nx+i);// x is the fastest dimension!!
		ksq=kx[i]+ky[j]+kz[k];
		chain->exp_ksq[ijk]=exp(-ds*ksq);
	                       }
                             }
                           }
	
	checkCudaErrors(cudaMallocManaged(&chain->exp_ksq_cu, sizeof(double)* NxNyNz));
	checkCudaErrors(cudaMallocManaged(&chemical->exp_w_cu, sizeof(double)* NxNyNz*chemical->N_spe));
	
	checkCudaErrors(cudaMemcpy(chain->exp_ksq_cu, chain->exp_ksq,sizeof(double)*NxNyNz,cudaMemcpyHostToDevice));

    checkCudaErrors(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
     

    // allocate chemical fields and density profile on CPU
    chemical->W_sp=dmatrix(0,chemical->N_spe-1,0,chemical->Nxyz-1);
    chemical->R_sp=dmatrix(0,chemical->N_spe-1,0,chemical->Nxyz-1);
    // propagators are allocated on GPU gobal memeory
    cudaEventRecord(start, 0);
    checkCudaErrors(cudaMallocManaged(&chain->qf, sizeof(double)* chain->Nxyz*chain->Ns));
    checkCudaErrors(cudaMallocManaged(&chain->qb, sizeof(double)* chain->Nxyz*chain->Ns));

    checkCudaErrors(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf ("Time for the kernel: %f ms\n", time);

}

extern void field_cp_gpu(GRID *grid,CHEMICAL *chemical,CHAIN *chain) {
     int Nx,Ny,Nz,i,j,k,spe;
     long ijk;
     Nx=grid->Nx;   
     Ny=grid->Ny;   
     Nz=grid->Nz;   
     double ds;
     ds=1.0/chain->Ns;
      for (spe=0;spe<chemical->N_spe;spe++) {
	for(k=0;k<Nz;k++) {
	   for(j=0;j<Ny;j++){
	      for(i=0;i<Nx;i++){
		ijk=(long)((k*Ny+j)*Nx+i+ spe*chemical->Nxyz );// x is the fastest dimension!!
		chemical->exp_w[ijk]=exp(-0.5*ds*chemical->W_sp[spe][ijk]);
	                       }
                             }
                           }
	                 }

    checkCudaErrors(cudaMemcpy(chemical->exp_w_cu, chemical->exp_w,sizeof(double)*chemical->Nxyz*chemical->N_spe,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

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



