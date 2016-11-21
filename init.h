#include"struct.h"
#include <helper_functions.h>
#include "init_cuda.h"
#include "field.h"

//void init_grid_cell(CUFFT_INFO *cufft_info,GPU_INFO *gpu_info, GRID *grid, CELL *cell);
//void init_scft(CUFFT_INFO *cufft_info,GPU_INFO *gpu_info, GRID *grid, CELL *cell, CHEMICAL *chemical, CHAIN *chain);
void init_AB_diblock_scft(CUFFT_INFO *cufft_info,GPU_INFO *gpu_info,GRID *grid, CELL *cell, CHEMICAL *chemical, CHAIN *chain);
void AB_diblock_scft_driver(CUFFT_INFO *cufft_info,GPU_INFO *gpu_info,GRID *grid, CELL *cell, CHEMICAL *chemical, CHAIN *chain );
