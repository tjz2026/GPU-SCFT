#ifndef	FIELD_H
#define FIELD_H
#include"struct.h"
// parameters for anderson mixing
extern int andsn_dim;
extern int simple_num;
extern int iter_counter;
extern double lambda_andsn; 
extern double ***andsn_W_sp;
extern double ***andsn_dW_sp;
extern double *global_t_diff;
extern double *yita;  // pressure field 

void field_init(int field_type, GRID *grid, CHEMICAL *chemical, CHAIN *chain );
//void andsn_init( int andsn_dim,int num, double lambda, grid &Grid, chemical &Chemical, chain &Chain );
//void andsn_iterate_diblock(int andsn_dim, grid &Grid, chemical &Chemical, chain &Chain, cell &Cell  ) ;
//int indx_update_andsn(int index_i, int n_r);
//void field_clean(CHEMICAL *chemical );

#endif
