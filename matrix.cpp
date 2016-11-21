///////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
void nrerror(char error_text[])
/*standard error handler */
{
	fprintf(stderr,"Numerical Recipes run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
	exit(1);
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
int *ivector_NR(long nl, long nh)
/* allocate an int vector with subscript range v[nl..nh] */
{
	int *v;
	v=(int *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(int)));
	if (!v) nrerror("allocation failure in ivector()");
	return v-nl+NR_END;
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
void free_ivector_NR(int *v, long nl, long nh)
/* free an int vector allocated with ivector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
double *dvector_NR(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
	double *v;
	v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
	if (!v) nrerror("allocation failure in dvector()");
	return v-nl+NR_END;
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
void free_dvector_NR(double *v, long nl, long nh)
/* free a double vector allocated with dvector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
int **imatrix_NR(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	int **m;
	
	/* allocate pointers to rows */
	m=(int **) malloc((size_t)((nrow+NR_END)*sizeof(int*)));
	if (!m) nrerror("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;
	
	/* allocate rows and set pointers to them */
	m[nrl]=(int *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(int)));
	if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;
	
	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
	
	/* return pointer to array of pointers to rows */
	return m;
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
void free_imatrix_NR(int **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by dmatrix() */
{
	free((FREE_ARG) (m[nrl]+ncl-NR_END));
	free((FREE_ARG) (m+nrl-NR_END));
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
double **dmatrix_NR(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	double **m;

	/* allocate pointers to rows */
	m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*)));
	if (!m) nrerror("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl]=(double *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double)));
	if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
void free_dmatrix_NR(double **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by dmatrix() */
{
	free((FREE_ARG) (m[nrl]+ncl-NR_END));
	free((FREE_ARG) (m+nrl-NR_END));
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
double *dvector(int nl, int nh)   
// for example:   dvector(0,NX-1)
/* allocate a double vector with subscript range v[nl..nh] */
{
	double *v;
	v=(double *)malloc((size_t) ((nh-nl+1)*sizeof(double)));
	if (!v) nrerror("allocation failure in dvector()");
	return v-nl;
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
void free_dvector(double *v, int nl, int nh)
/* free a double vector allocated with dvector() */
{
	free((FREE_ARG) (v+nl));
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
double **dmatrix(int nrl, int nrh, int ncl, int nch)   
// for example:   dmatrix(0,NX-1,0,NY-1)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	int i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	double **m;

	/* allocate pointers to rows */
	m=(double **) malloc((size_t)((nrow)*sizeof(double*)));
	if (!m) nrerror("allocation failure 1 in matrix()");
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl]=(double *) malloc((size_t)((nrow*ncol)*sizeof(double)));
	if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
void free_dmatrix(double **m, int nrl, int nrh, int ncl, int nch)
/* free a double matrix allocated by dmatrix() */
{
	free((FREE_ARG) (m[nrl]+ncl));
	free((FREE_ARG) (m+nrl));
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
double ***f3tensor(int nrl, int nrh, int ncl, int nch, int ndl, int ndh)   
// for example:  f3tensor(0,NX-1,0,NY-1,0,NZ-1)
/* allocate a double 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
	int i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
	double ***t;

	/* allocate pointers to pointers to rows */
	t=(double ***) malloc((size_t)((nrow)*sizeof(double**)));
	if (!t) nrerror("allocation failure 1 in f3tensor()");
	t -= nrl;

	/* allocate pointers to rows and set pointers to them */
	t[nrl]=(double **) malloc((size_t)((nrow*ncol)*sizeof(double*)));
	if (!t[nrl]) nrerror("allocation failure 2 in f3tensor()");
	t[nrl] -= ncl;

	/* allocate rows and set pointers to them */
	t[nrl][ncl]=(double *) malloc((size_t)((nrow*ncol*ndep)*sizeof(double)));
	if (!t[nrl][ncl]) nrerror("allocation failure 3 in f3tensor()");
	t[nrl][ncl] -= ndl;

	for(i=nrl+1;i<=nrh;i++) t[i]=t[i-1]+ncol;
        /////////////////////////////////////////////////////////////////////
	for(j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
	for(i=nrl+1;i<=nrh;i++) 
	{
		t[i][ncl]=t[i-1][ncl]+ncol*ndep;
	for(j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
	}

	/* return pointer to array of pointers to rows */
	return t;
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
void free_f3tensor(double ***t, int nrl, int nrh, int ncl, int nch,
	int ndl, int ndh)
/* free a double f3tensor allocated by f3tensor() */
{
	free((FREE_ARG) (t[nrl][ncl]+ndl));
	free((FREE_ARG) (t[nrl]+ncl));
	free((FREE_ARG) (t+nrl));
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
int ***f3tensor_int(int nrl, int nrh, int ncl, int nch, int ndl, int ndh)   
// for example:  f3tensor(0,NX-1,0,NY-1,0,NZ-1)
/* allocate a int 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
	int i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
	int ***t;
	
	/* allocate pointers to pointers to rows */
	t=(int ***) malloc((size_t)((nrow)*sizeof(int**)));
	if (!t) nrerror("allocation failure 1 in f3tensor()");
	t -= nrl;
	
	/* allocate pointers to rows and set pointers to them */
	t[nrl]=(int **) malloc((size_t)((nrow*ncol)*sizeof(int*)));
	if (!t[nrl]) nrerror("allocation failure 2 in f3tensor()");
	t[nrl] -= ncl;
	
	/* allocate rows and set pointers to them */
	t[nrl][ncl]=(int *) malloc((size_t)((nrow*ncol*ndep)*sizeof(int)));
	if (!t[nrl][ncl]) nrerror("allocation failure 3 in f3tensor()");
	t[nrl][ncl] -= ndl;
	
	for(i=nrl+1;i<=nrh;i++) t[i]=t[i-1]+ncol;
	/////////////////////////////////////////////////////////////////////
	for(j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
	for(i=nrl+1;i<=nrh;i++) 
	{
		t[i][ncl]=t[i-1][ncl]+ncol*ndep;
		for(j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
	}
	
	/* return pointer to array of pointers to rows */
	return t;
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
void free_f3tensor_int(int ***t, int nrl, int nrh, int ncl, int nch,
					   int ndl, int ndh)
					   /* free a int f3tensor allocated by f3tensor() */
{
	free((FREE_ARG) (t[nrl][ncl]+ndl));
	free((FREE_ARG) (t[nrl]+ncl));
	free((FREE_ARG) (t+nrl));
}
/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
