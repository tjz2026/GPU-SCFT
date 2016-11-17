
#include "struct.h"

template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

template <class T>
__global__ void
reduce3(T *g_idata, T *g_odata, unsigned int n)
{
	T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
	
	int block=blockIdx.x;
	unsigned int tid = threadIdx.x;
	
	T temp;
	T mySum = 0;
	
      for (unsigned int s=0; s<(n/blockDim.x+1); s++) {
	    temp=((s+threadIdx.x*(n/blockDim.x+1))<n)?g_idata[s+threadIdx.x*(n/blockDim.x+1)+block*n] : 0;
		
            mySum = mySum + temp;
    

	__syncthreads();
    }
    sdata[tid] = mySum;
	__syncthreads();
  mySum=0;
 	
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
		 
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }
	
	//printf("%d %g\n",tid,sdata[tid]);
    // write result for this block to global mem
    if (tid == 0) g_odata[block] = sdata[0];
	 __syncthreads();
}




__global__ void initilize_wdz(double *w,double *wdz,double ds2);

__global__ void initilize_q(double *q,double *qInt,int ns1);
	
__global__ void initilize_q_inverse(double *q,double *qInt,int ns1);

__global__ void initilize_in(double *in,double *g,double *wdz,int ns1,int iz);

__global__ void initilize_in_go(double *in,double *g,double *wdz,int ns1,int iz);

__global__ void sufaceField(cufftDoubleComplex *out,double *kxyzdz,int Nx);

__global__ void sufaceField_go(cufftDoubleComplex *out,double *kxyzdz,int Nxh1,int Nx,int Ny,int Nz);

__global__ void in_to_g(double *g,double *wdz_cu,cufftDoubleReal *in,int ns1,int iz);

__global__ void in_to_g_go(double *g,double *wdz_cu,cufftDoubleReal *in,int ns1,int iz);

__global__ void qInt_init(double *qInt);

__global__ void qa_to_qInt(double *qInt,double *qA,int NsA);

__global__ void qa_to_qInt2(double *qInt,double *qA,int NsA);

__global__ void cal_ql(double *ql_cu,double *qB_cu,int dNsB,int NxNyNz);

__global__ void w_to_phi(double *phlA, double *phlB,double *qA_cu,double *qcA_cu,double *qB_cu,double *qcB_cu,int NsA,int dNsB,double *ffl);

__global__ void w_to_phi_go(double *phlA, double *phlB,double *qA_cu,double *qcA_cu,double *qB_cu,double *qcB_cu,int NsA,int dNsB,double *ffl);

__global__ void minus_average(double *data,double *average_value);

__global__ void phi_w(double *wA_cu,double *wB_cu,double *phA_cu,double *phB_cu, double hAB);

__global__ void phi_w_constrained(double *wA_cu,double *wB_cu,double *phA_cu,double *phB_cu, double *PhA_cu,double hAB,double lambda);

__global__ void phi_w_constrainedEx(double *wA_cu,double *wB_cu,double *phA_cu,double *phB_cu, double *PhA_cu,double hAB);


