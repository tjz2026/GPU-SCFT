
#include <stdlib.h>

#include "struct.h"

#ifndef CUFFT
#define CUFFT



extern void init_cuda(GPU_INFO *gpu_info,int display);

extern void initialize_cufft(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info);

extern void finalize_cufft(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info);
extern void test_cufft(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info);


#endif
