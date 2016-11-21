#include"init.h"
#include "cuda_scft.h"
#include "scft_driver_AB_diblock.h"
#include<stdio.h>
/*
*/

void AB_diblock_scft_driver(CUFFT_INFO *cufft_info,GPU_INFO *gpu_info,GRID *grid, CELL *cell, CHEMICAL *chemical, CHAIN *chain ){
//   build the grid info and cufft info, allocate the memeory for scft arrays on both cpu and gpu
     init_AB_diblock_scft(cufft_info,gpu_info,grid, cell, chemical, chain);
// fill the fields with selected type, in this case, hexagonal is tested.
     const int Max_ITR=1000;
     const int ITR_print_frequency=50;
     const double error_tol=1.0e-5;
     int ITR;
     double error;
     for (ITR=0;ITR<Max_ITR;ITR++){ // SCFT iteration loop
         MDE_pesudo_spetrum();
         density_FE_cal();
         error=field_update();
         if (ITR%ITR_print_frequency ==0) {
         printf ("SCFT Iteration: error= %f at step %d\n", error,ITR);
                              }  
         if (error<error_tol) {
         printf ("SCFT converged, error= %f at step %d\n", error,ITR);
               break;   
                              }  

      }



}




