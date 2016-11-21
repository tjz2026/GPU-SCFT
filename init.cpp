#include"init.h"
#include<stdio.h>
/*
	initialize the grid and cell infomation
*/

void init_AB_diblock_scft(CUFFT_INFO *cufft_info,GPU_INFO *gpu_info,GRID *grid, CELL *cell, CHEMICAL *chemical, CHAIN *chain){
	
	char *typeChoice = 0;
	
	gpu_info->kernal_type=1;
	
	gpu_info->GPU_N=1; // specify the number of GPU to be used, otherwise by default is 1.
	char comment[200];
        int k,j;
	
	if(gpu_info->kernal_type==1){
	
		printf("GPU computing to be invoked.\n");
		
		FILE *fp;
		
		if((fp=fopen("Config.txt","r"))==false){
			printf("Configration file Config.txt doesn't exist, check it in your directory.\n");
		}

	
		fscanf(fp,"Dim=%d\n",&grid->Dim);
		fscanf(fp,"Nx=%d, Ny=%d, Nz=%d\n",&grid->Nx, &grid->Ny, &grid->Nz);
                cell->Cell_grid.Dim=grid->Dim;
                cell->Cell_grid.Nx=grid->Nx;
                cell->Cell_grid.Ny=grid->Ny;
                cell->Cell_grid.Nz=grid->Nz;
                GRID grid_tmp;
                grid_tmp.Dim=grid->Dim;  
                printf ("grid initialized.\n");  
		fscanf(fp,"lx=%lf, ly=%lf, lz=%lf\n",&cell->Lx, &cell->Ly, &cell->Lz);
		fclose(fp);
                cell->dx=cell->Lx/cell->Cell_grid.Nx;
                cell->dy=cell->Ly/cell->Cell_grid.Ny;
                cell->dz=cell->Lz/cell->Cell_grid.Nz;

                cufft_info->Nx=grid->Nx;  
                cufft_info->Ny=grid->Ny;  
                cufft_info->Nz=grid->Nz; 
                // currently only AB melts and AB diblock are supported in the program
                chemical->N_spe=2;
                for (k=0;k<chemical->N_spe;k++) {
                    for (j=0;j<chemical->N_spe;j++) {
                         chemical->XN[k][j]=0.0;
                                          } 
                                      }  
		fscanf(fp,"XN=%lf\n",&chemical->XN[0][1]);
                chemical->XN[1][0]=chemical->XN[0][1];
                chemical->Nxyz=grid->Nx*grid->Ny*grid->Nz;

                chain->N_blk=2; 
                chain->Blk_start=(int*)malloc(chain->N_blk*sizeof(int));
                chain->Blk_end=(int*)malloc(chain->N_blk*sizeof(int));
                chain->Blk_spe=(int*)malloc(chain->N_blk*sizeof(int));
                chain->Blk_f=(double*)malloc(chain->N_blk*sizeof(double));

                chain->Blk_spe[0]=0; // 0 for A, 1 for B
                chain->Blk_spe[1]=1; 
		fscanf(fp,"fA=%d\n",&chain->Blk_f[0]);
                chain->Blk_f[1]=1.0-chain->Blk_f[0];
		fscanf(fp,"Ns=%d\n",&chain->Ns);
		fclose(fp);

                int NA,NB;
                NA=int(chain->Ns*chain->Blk_f[0]);
                NB=chain->Ns-NA;
                chain->Blk_start[0]=0;
                chain->Blk_start[1]=NA;
                chain->Blk_end[0]=NA;
                chain->Blk_end[1]=chain->Ns;
                chain->Nxyz=chemical->Nxyz;  
		chain->Q=0.0;
		//! output configuration.
		
		printf("%d %d %d\n",cufft_info->Nx,cufft_info->Ny,cufft_info->Nz);	
		printf("lx=%lf, ly=%lf, lz=%lf\n",cell->Lx, cell->Ly, cell->Lz);
		
		init_cuda(gpu_info,0);
		
		initialize_cufft(gpu_info,cufft_info);
                test_cufft(gpu_info,cufft_info);
                // allocate the device memeory for chain and chemical;
                init_chain_chemical(gpu_info,grid,cell,chemical,chain);
                field_init(4,grid,chemical, chain);   // init hexagonal field
                field_cp_gpu(grid,chemical,chain); // copy field to gpu memeory  

                //finalize_cufft(gpu_info,cufft_info);
	
	          }
	
	
}






