#include"init.h"
#include<stdio.h>
/*
	initialize the grid and cell infomation
*/

void init_grid_cell(CUFFT_INFO *cufft_info,GPU_INFO *gpu_info,GRID *grid, CELL *cell){
	
	char *typeChoice = 0;
	
	gpu_info->kernal_type=1;
	
	gpu_info->GPU_N=1; // specify the number of GPU to be used, otherwise by default is 1.
	char comment[200];

	
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

		
		//! output configuration.
		
		printf("%d %d %d\n",cufft_info->Nx,cufft_info->Ny,cufft_info->Nz);	
		printf("lx=%lf, ly=%lf, lz=%lf\n",cell->Lx, cell->Ly, cell->Lz);
		
		
		
		init_cuda(gpu_info,0);
		
		initialize_cufft(gpu_info,cufft_info);
                test_cufft(gpu_info,cufft_info);
                
 //               finalize_cufft(gpu_info,cufft_info);
	
	          }
	
	
	
	
}
