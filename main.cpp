//  GPU-PSCF -GPU accelerated Polymer Self-Consistent Field Theory
//  Copyright (2016) Jiuzhou Tang
//  email: jiuzhou.tang@@theorie.physik.uni-goettingen.de
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation.
// ----------------------------------------------------------------------
//  File : main.cpp
//  PURPOSE
//    Main program for GPU-PSCF
//  AUTHOR 
//    Jiuzhou Tang
// --------

#include <stdio.h>

#include <assert.h>

#include<stdlib.h>

#include "struct.h" // data structure for scft program

#include "init_cuda.h"

#include "init.h"

#include <ctime>
// z dimension must be larger than Nz/GPU_N>=8
// CUDA runtime

int main(void){

	
	GPU_INFO gpu_info;
        GRID grid;   
        CELL cell;   
	CUFFT_INFO cufft_info;

	init_grid_cell(&cufft_info,&gpu_info,&grid,&cell);
	
//	fft_test(&gpu_info,&cufft_info);

	
	return 0;
}
