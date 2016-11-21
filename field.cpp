#include "field.h"
// initialize the field with certain specification.
int andsn_dim;
int simple_num;
int iter_counter;
double lambda_andsn; 
double ***andsn_W_sp;
double ***andsn_dW_sp;
double *global_t_diff;
double *yita;  // pressure field 

void field_init(int field_type, GRID *grid, CHEMICAL *chemical, CHAIN *chain )
{

   int i1,j1,k1;
   long K;
   double temp_coff=0.8;
// init the pressure field
// select the initial field type
switch (field_type) {
// random noise 
   case 0 : {
     break;
         }
// FCC
   case 1 : {
  
     break;
         }
// BCC
   case 2 : {
             
     break;
         }
// GYR
   case 3 : {

     break;
         }
// HEX
   case 4 : {
    assert(grid->Dim==2 || grid->Dim==3);
// the hexagonal structure is kept translation constant along the z axis, and the ratio of size on X and Y axis is sqrt(3)         
      for (i1=0,K=0;i1<=grid->Nz;i1++) {  // Z axis
         for (j1=0;j1<=grid->Ny;j1++)      { // Y axis
            for (k1=0;k1<=grid->Nx;k1++,K++) { // X axis
                chemical->R_sp[0][K]=(chain->Blk_f[0])*(1.0+ \ 
                 temp_coff*cos(2.0*Pi*(j1+1)/grid->Ny)*cos(2.0*Pi*(k1+1)/grid->Nx));
                                                              }
                                                           }

                                                       }    
     break;
         }
// LAM
   case 5 : {
// the sine-like lam is  along the X axis
      for (i1=0,K=0;i1<=grid->Nz;i1++) {  // Z axis
         for (j1=0;j1<=grid->Ny;j1++)      { // Y axis
            for (k1=0;k1<=grid->Nx;k1++,K++) { // X axis
                chemical->R_sp[0][K]=(chain->Blk_f[0])*(1.0+ \ 
                 temp_coff*sin(2.0*Pi*(k1+1)/grid->Nx));
                                                              }
                                                           }
                                                       }    

  
     break;
         }

   default : {}
     break;
 } 

// currently, only AB melts is supported, field initialization of more molecular species is to be implemented.
      for (i1=0,K=0;i1<=grid->Nz;i1++) {  // Z axis
         for (j1=0;j1<=grid->Ny;j1++)      { // Y axis
            for (k1=0;k1<=grid->Nx;k1++,K++) { // X axis
                chemical->R_sp[1][K]=1.0-chemical->R_sp[0][K];
                chemical->W_sp[0][K]=chemical->XN[0][1]*chemical->R_sp[1][K];
                chemical->W_sp[1][K]=chemical->XN[0][1]*chemical->R_sp[0][K];
                                                              }
                                                           }
                                                       }    

}

