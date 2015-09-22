/* 
*  Copyright (c) 2008  Noah Snavely (snavely (at) cs.washington.edu)
*    and the University of Washington
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*/

/* 5point.c */
/* Solve the 5-point relative pose problem */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "defines.h"
#include "fmatrix.h"
#include "poly1.h"
#include "poly3.h"
#include "matrix.h"
#include "qsort.h"
#include "svd.h"
#include "triangulate.h"
#include "vector.h"

#include <opencv/cv.h>

void matrix_print_f(int m, int n, double *A) {
    int i, j;
    
    for (i = 0; i < m; i++) {
        printf("  ");
        for (j = 0; j < n; j++) {
            printf(" %0.6f ", A[i * n + j]);
        }
        printf("\n");
    }
}

int find_extrinsics_essential_multipt_depth_support(double *E, int n,
													v2_t *p1, v2_t *p2, v3_t *p1_3d, v3_t *p2_3d, 
													double *R, double *t, int *inlierList)
{
	double tmp[9], tmp2[3], Qv[3], PQ_tv[3];
	double U[9], S[3], VT[9];
	double tu[3], Ra[9], Rb[9], tt[3];
	double scalex = 0, scaley = 0, scalez = 0;
	double cntScale = 0, cntPoint = 0;

	double D[9] = 
	{  0.0, 1.0, 0.0,
	-1.0, 0.0, 0.0,
	0.0, 0.0, 1.0 };

	double DT[9] = 
	{  0.0, -1.0, 0.0,
	1.0,  0.0, 0.0,
	0.0,  0.0, 1.0 };

	double I[9] = 
	{  1.0, 0.0, 0.0, 
	0.0, 1.0, 0.0,
	0.0, 0.0, 1.0 };

	double t0[3] = { 0.0, 0.0, 0.0 };

	v3_t Q, PQ, PQ_t, Qt;
	double c1, c2;
	double error;

	int i;
	int c1_pos = 0, c1_neg = 0;
	int c2_pos = 0, c2_neg = 0;

	/* Now find the SVD of E */
	dgesvd_driver(3, 3, E, U, S, VT);

#if 0
	printf("S[0] = %0.3e\n", S[0]);
	printf("S[1] = %0.3e\n", S[1]);
	printf("S[2] = %0.3e\n", S[2]);
#endif

	/* Now find R and t */
	tu[0] = U[2];  tu[1] = U[5];  tu[2] = U[8];

	matrix_product33(U, D, tmp);
	matrix_product33(tmp, VT, Ra);
	matrix_product33(U, DT, tmp);

	matrix_product33(tmp, VT, Rb);

	if (matrix_determinant3(Ra) < 0.0) {
		// printf("flipping...\n");
		matrix_scale(3, 3, Ra, -1.0, Ra);
	}

	if (matrix_determinant3(Rb) < 0.0) {
		// printf("flopping...\n");
		matrix_scale(3, 3, Rb, -1.0, Rb);
	}

	printf("Ra = \n");
	matrix_print_f(3, 3, Ra);
	printf("Rb = \n");
	matrix_print_f(3, 3, Rb);
	printf("tu = \n");
	matrix_print_f(1, 3, tu);

	/* Figure out which configuration is correct using the supplied
	* points */
	// R * p1 + t = p2
	for (i = 0; i < n; i++) {
		if( inlierList[i] < 1 )continue;
		Q = p1_3d[i]; PQ_t = p2_3d[i];
		if( -Vz(Q) < 0.01 || -Vz(Q) > 3.0 || -Vz(PQ_t) < 0.01 || -Vz(PQ_t) > 3.0 ) continue;
		cntPoint ++;
		Qv[0] = Vx(Q), Qv[1] = Vy(Q), Qv[2] = -Vz(Q);

		//Qt = triangulate(p1[i], p2[i], I, t0, Ra, tu, &error);
		//printf("3dpoint = \n tr: %0.3f, %0.3f, %0.3f \n p1: %0.3f, %0.3f, %0.3f \n p2: %0.3f, %0.3f, %0.3f \n", 
		//	Vx(Qt), Vy(Qt), Vz(Qt), Vx(Q), Vy(Q), Vz(Q), Vx(PQ_t), Vy(PQ_t), Vz(PQ_t));


		PQ_tv[0] = Vx(PQ_t), PQ_tv[1] = Vy(PQ_t), PQ_tv[2] = -Vz(PQ_t);

		matrix_product331(Ra, Qv, tmp); // tmp = R * Q
		matrix_cross(tmp, PQ_tv, tmp2); // check if R * Q parrallel with PQ_true (i.e. R*Q + t == PQ_true)
		if( matrix_normsq(3, 1, tmp2) < 0.1 ) { // if parrallel; t = PQ_true - R * Q;
			tt[0] = PQ_tv[0] - tmp[0]; tt[1] = PQ_tv[1] - tmp[1]; tt[2] = PQ_tv[2] - tmp[2];
			//scale = scale + ( (tt[0]/tu[0]) + (tt[1]/tu[1]) + (tt[2]/tu[2]) );
			//scale = scale + ( (tt[0]+tt[1]+tt[2])/(tu[0]+tu[1]+tu[2]) );
			scalex += (tt[0]/tu[0]); scaley += (tt[1]/tu[1]); scalez += (tt[2]/tu[2]);
			cntScale ++;
		}
	}

	if (cntScale / cntPoint > 0.5) { // if Ra is right, return R and t;
		memcpy(R, Ra, sizeof(double)*9);
		scalex = scalex / cntScale; scaley = scaley / cntScale; scalez = scalez / cntScale;
		t[0] = tu[0] * scalex; t[1] = tu[1] * scaley; t[2] = tu[2] * scalez;
		printf("Ra Ratio = %0.3f / %0.3f ; scale = (%0.3f, %0.3f, %0.3f)\n", cntScale, cntPoint, scalex, scaley, scalez);
	}
	else { // Ra is wrong, check Rb
		cntPoint = 0; cntScale = 0; scalex = scaley = scalez = 0;
		for (i = 0; i < n; i ++) {
			if( inlierList[i] < 1 )continue;
			Q = p1_3d[i]; PQ_t = p2_3d[i];
			if( -Vz(Q) < 0.01 || -Vz(Q) > 3.0 || -Vz(PQ_t) < 0.01 || -Vz(PQ_t) > 3.0 ) continue;
			cntPoint ++;
			Qv[0] = Vx(Q), Qv[1] = Vy(Q), Qv[2] = -Vz(Q);
			PQ_tv[0] = Vx(PQ_t), PQ_tv[1] = Vy(PQ_t), PQ_tv[2] = -Vz(PQ_t);

			//Qt = triangulate(p1[i], p2[i], I, t0, Ra, tu, &error);
			//printf("3dpoint = \n tr: %0.3f, %0.3f, %0.3f \n p1: %0.3f, %0.3f, %0.3f \n p2: %0.3f, %0.3f, %0.3f \n", 
			//Vx(Qt), Vy(Qt), Vz(Qt), Vx(Q), Vy(Q), Vz(Q), Vx(PQ_t), Vy(PQ_t), Vz(PQ_t));

			matrix_product331(Rb, Qv, tmp); // tmp = R * Q
			matrix_cross(tmp, PQ_tv, tmp2); // check if R * Q parrallel with PQ_true (i.e. R*Q + t == PQ_true)
			if( matrix_normsq(3, 1, tmp2) < 0.1 ) { // if parrallel; t = PQ_true - R * Q;
				tt[0] = PQ_tv[0] - tmp[0]; tt[1] = PQ_tv[1] - tmp[1]; tt[2] = PQ_tv[2] - tmp[2];
				//scale = scale + ( (tt[0]/tu[0]) + (tt[1]/tu[1]) + (tt[2]/tu[2]) );
				//scale = scale + ( (tt[0]+tt[1]+tt[2])/(tu[0]+tu[1]+tu[2]) );
				scalex += (tt[0]/tu[0]); scaley += (tt[1]/tu[1]); scalez += (tt[2]/tu[2]);
				cntScale ++;
			}
		}

		if (cntScale / cntPoint > 0.5) { // if Ra is right, return R and t;
			memcpy(R, Rb, sizeof(double)*9);
					scalex = scalex / cntScale; scaley = scaley / cntScale; scalez = scalez / cntScale;
		t[0] = tu[0] * scalex; t[1] = tu[1] * scaley; t[2] = tu[2] * scalez;
		printf("Rb Ratio = %0.3f / %0.3f ; scale = (%0.3f, %0.3f, %0.3f)\n", cntScale, cntPoint, scalex, scaley, scalez);
		}
		else {
			//printf("Ra = \n");
			//matrix_print_f(3, 3, Ra);
			//printf("Rb = \n");
			//matrix_print_f(3, 3, Rb);
			//printf("tu = \n");
			//matrix_print_f(1, 3, tu);
			fprintf(stderr, "[find_extrinsics] Error: no case found!\n");
			return 0;
		}
	}

	return 1;
}


void compute_nullspace_basis(int n, v2_t *a, v2_t *b, double *basis) {
	double *Q, *S, *U, VT[81];
	int max_dim;
	int i;

	if (n < 5) {
		fprintf(stderr, "[compute_nullspace_basis] n must be >= 5\n");
		return;
	}

	max_dim = MAX(9, n);

	Q = malloc(sizeof(double) * max_dim * max_dim);
	S = malloc(sizeof(double) * max_dim);
	U = malloc(sizeof(double) * max_dim * max_dim);


	/* Create the 5x9 epipolar constraint matrix */
	for (i = 0; i < 5; i++) {
		double *row = Q + i * 9;

		row[0] = a[i].p[0] * b[i].p[0];
		row[1] = a[i].p[1] * b[i].p[0];
		row[2] = b[i].p[0];

		row[3] = a[i].p[0] * b[i].p[1];
		row[4] = a[i].p[1] * b[i].p[1];
		row[5] = b[i].p[1];

		row[6] = a[i].p[0];
		row[7] = a[i].p[1];
		row[8] = 1.0;
	}

	/* Find four vectors that span the right nullspace of the matrix */
	dgesvd_driver(n, 9, Q, U, S, VT);

	memcpy(basis, VT + 5 * 9, 36 * sizeof(double));

	free(Q);
	free(S);
	free(U);
}

void compute_constraint_matrix(double *basis, poly3_t *constraints)
{
	/* Basis rows are X, Y, Z, W 
	* Essential matrix is or form x*X + y*Y + z*Z + W */

	/* Create a polynomial for each entry of E */
	poly3_t polys[9];
	poly3_t poly_term1, poly_term2, poly_term3, poly_det;
	poly3_t poly_EET[6], poly_lambda[6], poly_tr, poly_lambdaE[9];

	int i;

	for (i=0; i < 9; i++) {
		polys[i] = poly3_new(basis[i], basis[9+i], basis[18+i], basis[27+i]);
	}

	/* Create a polynormial from the constraint det(E) = 0 */
	poly_term1 = poly3_mult21( poly3_sub( poly3_mult11(polys[1], polys[5]), 
		poly3_mult11(polys[2], polys[4]) ), 
		polys[6] );

	// matrix_print(1, 20, poly_term1.v);

	poly_term2 = poly3_mult21( poly3_sub( poly3_mult11(polys[2], polys[3]), 
		poly3_mult11(polys[0], polys[5]) ), 
		polys[7] );

	poly_term3 = poly3_mult21( poly3_sub( poly3_mult11(polys[0], polys[4]), 
		poly3_mult11(polys[1], polys[3]) ), 
		polys[8] );

	poly_det = poly3_add(poly_term1, poly3_add(poly_term2, poly_term3));

	/* Create polynomials for the singular value constraint */
	for (i = 0; i < 6; i++) {
		int r = 0, c = 0, k;
		poly_EET[i] = poly3_new(0.0, 0.0, 0.0, 0.0);
		switch(i) {
		case 0:
		case 1:
		case 2:
			r = 0;
			c = i;
			break;
		case 3:
		case 4:
			r = 1;
			c = i-2;
			break;
		case 5:
			r = 2;
			c = 2;
			break;
		}

		for (k = 0; k < 3; k++) {
			poly_EET[i] = poly3_add(poly_EET[i], poly3_mult11(polys[r*3+k], polys[c*3+k]));
		}
	}

	poly_tr = poly3_add3(poly_EET[0], poly_EET[3], poly_EET[5]);
	poly_tr = poly3_scale(poly_tr, 0.5);

	poly_lambda[0] = poly3_sub(poly_EET[0], poly_tr);
	poly_lambda[1] = poly_EET[1];
	poly_lambda[2] = poly_EET[2];

	poly_lambda[3] = poly3_sub(poly_EET[3], poly_tr);
	poly_lambda[4] = poly_EET[4];

	poly_lambda[5] = poly3_sub(poly_EET[5], poly_tr);

	poly_lambdaE[0] = poly3_add3(poly3_mult(poly_lambda[0], polys[0]),
		poly3_mult(poly_lambda[1], polys[3]),
		poly3_mult(poly_lambda[2], polys[6]));

	poly_lambdaE[1] = poly3_add3(poly3_mult(poly_lambda[0], polys[1]),
		poly3_mult(poly_lambda[1], polys[4]),
		poly3_mult(poly_lambda[2], polys[7]));

	poly_lambdaE[2] = poly3_add3(poly3_mult(poly_lambda[0], polys[2]),
		poly3_mult(poly_lambda[1], polys[5]),
		poly3_mult(poly_lambda[2], polys[8]));

	poly_lambdaE[3] = poly3_add3(poly3_mult21(poly_lambda[1], polys[0]),
		poly3_mult21(poly_lambda[3], polys[3]),
		poly3_mult21(poly_lambda[4], polys[6]));

	poly_lambdaE[4] = poly3_add3(poly3_mult21(poly_lambda[1], polys[1]),
		poly3_mult21(poly_lambda[3], polys[4]),
		poly3_mult21(poly_lambda[4], polys[7]));

	poly_lambdaE[5] = poly3_add3(poly3_mult21(poly_lambda[1], polys[2]),
		poly3_mult21(poly_lambda[3], polys[5]),
		poly3_mult21(poly_lambda[4], polys[8]));

	poly_lambdaE[6] = poly3_add3(poly3_mult21(poly_lambda[2], polys[0]),
		poly3_mult21(poly_lambda[4], polys[3]),
		poly3_mult21(poly_lambda[5], polys[6]));

	poly_lambdaE[7] = poly3_add3(poly3_mult21(poly_lambda[2], polys[1]),
		poly3_mult21(poly_lambda[4], polys[4]),
		poly3_mult21(poly_lambda[5], polys[7]));

	poly_lambdaE[8] = poly3_add3(poly3_mult21(poly_lambda[2], polys[2]),
		poly3_mult21(poly_lambda[4], polys[5]),
		poly3_mult21(poly_lambda[5], polys[8]));

	for (i=0; i < 9; i++) 
		constraints[i] = poly_lambdaE[i];

	constraints[9] = poly_det;
}

void eliminate_gauss_jordan(poly3_t *constraints)
{
	double A[200];
	int i, j;

	for (i = 0; i < 10; i++) {
		// memcpy(A + 20 * i, constraints[i].v, sizeof(double) * 20);
		double *row = A + 20 * i;
		row[0] = constraints[i].v[POLY3_X3];
		row[1] = constraints[i].v[POLY3_Y3];
		row[2] = constraints[i].v[POLY3_X2Y];
		row[3] = constraints[i].v[POLY3_XY2];
		row[4] = constraints[i].v[POLY3_X2Z];
		row[5] = constraints[i].v[POLY3_X2];
		row[6] = constraints[i].v[POLY3_Y2Z];
		row[7] = constraints[i].v[POLY3_Y2];
		row[8] = constraints[i].v[POLY3_XYZ];
		row[9] = constraints[i].v[POLY3_XY];
		row[10] = constraints[i].v[POLY3_XZ2];
		row[11] = constraints[i].v[POLY3_XZ];
		row[12] = constraints[i].v[POLY3_X];
		row[13] = constraints[i].v[POLY3_YZ2];
		row[14] = constraints[i].v[POLY3_YZ];
		row[15] = constraints[i].v[POLY3_Y];
		row[16] = constraints[i].v[POLY3_Z3];
		row[17] = constraints[i].v[POLY3_Z2];
		row[18] = constraints[i].v[POLY3_Z];
		row[19] = constraints[i].v[POLY3_UNIT];
	}

	for (i = 0; i < 10; i++) {
		/* Make the leading coefficient of row i = 1 */
		double leading = A[20 * i + i];
		matrix_scale(20, 1, A + 20 * i, 1.0 / leading, A + 20 * i);

		/* Subtract from other rows */
		for (j = i+1; j < 10; j++) {
			double leading2 = A[20 * j + i];
			double scaled_row[20];
			matrix_scale(20, 1, A + 20 * i, leading2, scaled_row);
			matrix_diff(20, 1, 20, 1, A + 20 * j, scaled_row, A + 20 * j);
		}
	}

	/* Now, do the back substitution (stopping four rows early) */
	for (i = 9; i >= 4; i--) {
		for (j = 0; j < i; j++) {
			double scale = A[20 * j + i];
			double scaled_row[20];
			matrix_scale(20, 1, A + 20 * i, scale, scaled_row);
			matrix_diff(20, 1, 20, 1, A + 20 * j, scaled_row, A + 20 * j);
		}
	}

	/* Copy out results */
	for (i = 0; i < 10; i++) {
		// memcpy(constraints[i].v, A + 20 * i, sizeof(double) * 20);
		double *row = A + 20 * i;
		constraints[i].v[POLY3_X3] = row[0];
		constraints[i].v[POLY3_Y3] = row[1];
		constraints[i].v[POLY3_X2Y] = row[2];
		constraints[i].v[POLY3_XY2] = row[3];
		constraints[i].v[POLY3_X2Z] = row[4];
		constraints[i].v[POLY3_X2] = row[5];
		constraints[i].v[POLY3_Y2Z] = row[6];
		constraints[i].v[POLY3_Y2] = row[7];
		constraints[i].v[POLY3_XYZ] = row[8];
		constraints[i].v[POLY3_XY] = row[9];
		constraints[i].v[POLY3_XZ2] = row[10];
		constraints[i].v[POLY3_XZ] = row[11];
		constraints[i].v[POLY3_X] = row[12];
		constraints[i].v[POLY3_YZ2] = row[13];
		constraints[i].v[POLY3_YZ] = row[14];
		constraints[i].v[POLY3_Y] = row[15];
		constraints[i].v[POLY3_Z3] = row[16];
		constraints[i].v[POLY3_Z2] = row[17];
		constraints[i].v[POLY3_Z] = row[18];
		constraints[i].v[POLY3_UNIT] = row[19];
	}
}

void compute_B_matrix(poly3_t *constraints, poly1_t *B) 
{
	poly3_t e = constraints[4];
	poly3_t f = constraints[5];
	poly3_t g = constraints[6];
	poly3_t h = constraints[7];
	poly3_t i = constraints[8];
	poly3_t j = constraints[9];

	B[0] = poly1_new3(-poly3_get(f, POLY3_XZ2),
		poly3_get(e, POLY3_XZ2) - poly3_get(f, POLY3_XZ),
		poly3_get(e, POLY3_XZ) - poly3_get(f, POLY3_X),
		poly3_get(e, POLY3_X));

	B[1] = poly1_new3(-poly3_get(f, POLY3_YZ2),
		poly3_get(e, POLY3_YZ2) - poly3_get(f, POLY3_YZ),
		poly3_get(e, POLY3_YZ) - poly3_get(f, POLY3_Y),
		poly3_get(e, POLY3_Y));

	B[2] = poly1_new4(-poly3_get(f, POLY3_Z3),
		poly3_get(e, POLY3_Z3) - poly3_get(f, POLY3_Z2),
		poly3_get(e, POLY3_Z2) - poly3_get(f, POLY3_Z),
		poly3_get(e, POLY3_Z) - poly3_get(f, POLY3_UNIT),
		poly3_get(e, POLY3_UNIT));

	B[3] = poly1_new3(-poly3_get(h, POLY3_XZ2),
		poly3_get(g, POLY3_XZ2) - poly3_get(h, POLY3_XZ),
		poly3_get(g, POLY3_XZ) - poly3_get(h, POLY3_X),
		poly3_get(g, POLY3_X));

	B[4] = poly1_new3(-poly3_get(h, POLY3_YZ2),
		poly3_get(g, POLY3_YZ2) - poly3_get(h, POLY3_YZ),
		poly3_get(g, POLY3_YZ) - poly3_get(h, POLY3_Y),
		poly3_get(g, POLY3_Y));

	B[5] = poly1_new4(-poly3_get(h, POLY3_Z3),
		poly3_get(g, POLY3_Z3) - poly3_get(h, POLY3_Z2),
		poly3_get(g, POLY3_Z2) - poly3_get(h, POLY3_Z),
		poly3_get(g, POLY3_Z) - poly3_get(h, POLY3_UNIT),
		poly3_get(g, POLY3_UNIT));

	B[6] = poly1_new3(-poly3_get(j, POLY3_XZ2),
		poly3_get(i, POLY3_XZ2) - poly3_get(j, POLY3_XZ),
		poly3_get(i, POLY3_XZ) - poly3_get(j, POLY3_X),
		poly3_get(i, POLY3_X));

	B[7] = poly1_new3(-poly3_get(j, POLY3_YZ2),
		poly3_get(i, POLY3_YZ2) - poly3_get(j, POLY3_YZ),
		poly3_get(i, POLY3_YZ) - poly3_get(j, POLY3_Y),
		poly3_get(i, POLY3_Y));

	B[8] = poly1_new4(-poly3_get(j, POLY3_Z3),
		poly3_get(i, POLY3_Z3) - poly3_get(j, POLY3_Z2),
		poly3_get(i, POLY3_Z2) - poly3_get(j, POLY3_Z),
		poly3_get(i, POLY3_Z) - poly3_get(j, POLY3_UNIT),
		poly3_get(i, POLY3_UNIT));
}

void compute_determinant(poly1_t *B, poly1_t *p1, poly1_t *p2, poly1_t *p3, 
						 poly1_t *det)
{
	*p1 = poly1_sub( poly1_mult( B[1], B[5] ), 
		poly1_mult( B[2], B[4] ) );

	*p2 = poly1_sub( poly1_mult( B[2], B[3] ), 
		poly1_mult( B[0], B[5] ) ); 

	*p3 = poly1_sub( poly1_mult( B[0], B[4] ), 
		poly1_mult( B[1], B[3] ) );

	*det = poly1_add3(poly1_mult(*p1, B[6]), 
		poly1_mult(*p2, B[7]),
		poly1_mult(*p3, B[8]));
}

void extract_roots(poly1_t det, int *num_roots, double *roots)
{
	double C[100], evec[100], eval[10];

	/* Scale the determinant */
	poly1_t det_scale = poly1_normalize(det);

	int i, real;

	/* Fill the companion matrix */
	for (i = 0; i < 100; i++)
		C[i] = 0.0;

	for (i = 0; i < 10; i++) 
		C[i] = -det_scale.v[9-i];

	for (i = 1; i < 10; i++)
		C[i * 10 + (i-1)] = 1.0;

	real = dgeev_driver(10, C, evec, eval);

	memcpy(roots, eval, real * sizeof(double));
	*num_roots = real;
}

void compute_Ematrices(int n, double *roots, double *basis, 
					   poly1_t p1, poly1_t p2, poly1_t p3, double *E) {
						   int i;

						   for (i = 0; i < n; i++) {
							   double z = roots[i];
							   double den = poly1_eval(p3, z);
							   double den_inv = 1.0 / den;

							   double x = poly1_eval(p1, z) * den_inv;
							   double y = poly1_eval(p2, z) * den_inv;

							   double X[9], Y[9], Z[9], tmp1[9], tmp2[9];

							   matrix_scale(9, 1, basis + 0, x, X);
							   matrix_scale(9, 1, basis + 9, y, Y);
							   matrix_scale(9, 1, basis + 18, z, Z);

							   matrix_sum(9, 1, 9, 1, X, Y, tmp1);
							   matrix_sum(9, 1, 9, 1, Z, basis + 27, tmp2);
							   matrix_sum(9, 1, 9, 1, tmp1, tmp2, E + 9 * i);
						   }
}


void compute_Grabner_basis(poly3_t *constraints, double *Gbasis) 
{
	double A[200];
	int i, j;

	for (i = 0; i < 10; i++) {
		// memcpy(A + 20 * i, constraints[i].v, sizeof(double) * 20);
		double *row = A + 20 * i;

		/* x3 x2y xy2 y3 x2z xyz y2z xz2 yz2 z3 x2 xy y2 xz yz z2 x  y  z  1 */

		row[0] = constraints[i].v[POLY3_X3];
		row[1] = constraints[i].v[POLY3_X2Y];
		row[2] = constraints[i].v[POLY3_XY2];
		row[3] = constraints[i].v[POLY3_Y3];
		row[4] = constraints[i].v[POLY3_X2Z];
		row[5] = constraints[i].v[POLY3_XYZ];
		row[6] = constraints[i].v[POLY3_Y2Z];
		row[7] = constraints[i].v[POLY3_XZ2];
		row[8] = constraints[i].v[POLY3_YZ2];
		row[9] = constraints[i].v[POLY3_Z3];
		row[10] = constraints[i].v[POLY3_X2];
		row[11] = constraints[i].v[POLY3_XY];
		row[12] = constraints[i].v[POLY3_Y2];
		row[13] = constraints[i].v[POLY3_XZ];
		row[14] = constraints[i].v[POLY3_YZ];
		row[15] = constraints[i].v[POLY3_Z2];
		row[16] = constraints[i].v[POLY3_X];
		row[17] = constraints[i].v[POLY3_Y];
		row[18] = constraints[i].v[POLY3_Z];
		row[19] = constraints[i].v[POLY3_UNIT];
	}

	/* Do a full Gaussian elimination */

	for (i = 0; i < 10; i++) {
		/* Make the leading coefficient of row i = 1 */
		double leading = A[20 * i + i];
		matrix_scale(20, 1, A + 20 * i, 1.0 / leading, A + 20 * i);

		/* Subtract from other rows */
		for (j = i+1; j < 10; j++) {
			double leading2 = A[20 * j + i];
			double scaled_row[20];
			matrix_scale(20, 1, A + 20 * i, leading2, scaled_row);
			matrix_diff(20, 1, 20, 1, A + 20 * j, scaled_row, A + 20 * j);
		}
	}

	/* Now, do the back substitution */
	for (i = 9; i >= 0; i--) {
		for (j = 0; j < i; j++) {
			double scale = A[20 * j + i];
			double scaled_row[20];
			matrix_scale(20, 1, A + 20 * i, scale, scaled_row);
			matrix_diff(20, 1, 20, 1, A + 20 * j, scaled_row, A + 20 * j);
		}
	}

	/* Copy out results */
	for (i = 0; i < 10; i++) {
		memcpy(Gbasis + i * 10, A + i * 20 + 10, sizeof(double) * 10);
	}
}

void compute_action_matrix(double *Gbasis, double *At) 
{
	int i;
	for (i = 0; i < 100; i++)
		At[i] = 0.0;

	matrix_scale(10, 1, Gbasis +  0, -1.0, At +  0);
	matrix_scale(10, 1, Gbasis + 10, -1.0, At + 10);
	matrix_scale(10, 1, Gbasis + 20, -1.0, At + 20);
	matrix_scale(10, 1, Gbasis + 40, -1.0, At + 30);
	matrix_scale(10, 1, Gbasis + 50, -1.0, At + 40);
	matrix_scale(10, 1, Gbasis + 70, -1.0, At + 50);

	At[6 * 10 + 0] = 1.0;
	At[7 * 10 + 1] = 1.0;
	At[8 * 10 + 3] = 1.0;
	At[9 * 10 + 6] = 1.0;
}

void compute_Ematrices_Gb(double *At, double *basis, int *num_solns, double *E)
{
	double evec[100], eval[10];
	int real = dgeev_driver(10, At, evec, eval);
	int i;

	for (i = 0; i < real; i++) {
		double X[9], Y[9], Z[9], tmp1[9], tmp2[9];

		double x = evec[10 * i + 6];
		double y = evec[10 * i + 7];
		double z = evec[10 * i + 8];
		double w = evec[10 * i + 9];
		double w_inv = 1.0 / w;

		x = x * w_inv;
		y = y * w_inv;
		z = z * w_inv;

		matrix_scale(9, 1, basis + 0, x, X);
		matrix_scale(9, 1, basis + 9, y, Y);
		matrix_scale(9, 1, basis + 18, z, Z);

		matrix_sum(9, 1, 9, 1, X, Y, tmp1);
		matrix_sum(9, 1, 9, 1, Z, basis + 27, tmp2);
		matrix_sum(9, 1, 9, 1, tmp1, tmp2, E + 9 * i);

		matrix_scale(9, 1, E + 9 * i, 1.0 / E[9 * i + 8], E + 9 * i);
	}

	*num_solns = real;
}

void generate_Ematrix_hypotheses(int n, v2_t *rt_pts, v2_t *left_pts, 
								 int *num_poses, double *E) 
{
	double basis[36], Gbasis[100], At[100];
	poly3_t constraints[10];
	// poly1_t B[9], p1, p2, p3, det;
	// double roots[10];

	if (n < 5) {
		fprintf(stderr, "[generate_Ematrix_hypotheses] n must be >= 5\n");
		return;
	}

	/* Generate the nullspace basis of the epipolar constraint matrix */
	compute_nullspace_basis(n, rt_pts, left_pts, basis);
	compute_constraint_matrix(basis, constraints);

#if 0
	eliminate_gauss_jordan(constraints);
	compute_B_matrix(constraints, B);
	compute_determinant(B, &p1, &p2, &p3, &det);
	extract_roots(det, num_poses, roots);
	compute_Ematrices(*num_poses, roots, basis, p1, p2, p3, E);
#else
	compute_Grabner_basis(constraints, Gbasis);
	compute_action_matrix(Gbasis, At);
	compute_Ematrices_Gb(At, basis, num_poses, E);
#endif
}

void choose(int n, int k, int *arr)
{
	int i;

	if (k > n) {
		fprintf(stderr, "[choose] Error: k > n\n");
		return;
	}

	for (i = 0; i < k; i++) {
		while (1) {
			int idx = rand() % n;
			int j, redo = 0;

			for (j = 0; j < i; j++) {
				if (idx == arr[j]) {
					redo = 1;
					break;
				}
			}

			if (!redo) {
				arr[i] = idx;
				break;
			}
		}
	}
}

int evaluate_Ematrix(int n, v2_t *r_pts, v2_t *l_pts, double thresh_norm,
					 double *F, int *best_inlier, double *score, int *inlierList)
{
	int num_inliers = 0;
	int i;
	double min_resid = 1.0e20;
	double likelihood = 0.0;
	//int cnt = 0;
	for (i = 0; i < n; i++) {
#if 1
		v3_t r = v3_new(Vx(r_pts[i]), Vy(r_pts[i]), 1.0);
		v3_t l = v3_new(Vx(l_pts[i]), Vy(l_pts[i]), 1.0);
#else
		v3_t r = v3_new(-Vx(r_pts[i]), -Vy(r_pts[i]), 1.0);
		v3_t l = v3_new(-Vx(l_pts[i]), -Vy(l_pts[i]), 1.0);
#endif

#if 0
		double resid = fmatrix_compute_residual(E, l, r);
#else
		double resid = fmatrix_compute_residual(F, l, r);
#endif     

		likelihood += log(1.0 + resid * resid / (thresh_norm));

		if (resid < thresh_norm) {
			num_inliers++;
			inlierList[i] = 1;
			if (resid < min_resid) {
				min_resid = resid;
				*best_inlier = i;
			}
		}
	}

	*score = likelihood;
	// *score = 1.0 / num_inliers;

	return num_inliers;
}

void compute_t_ransac( int n, double * tx, double * ty, double * tz, double * t ) {
	double thresh = 0.05;
	double scale[3];
	int inlierNum = 0, inlierNumMax = 0;

	int i, j;
	for(i = 0; i < n; i ++) {
		inlierNum = 0;
		for(j = 0; j < 3; j ++) scale[j] = 0;
		for(j = 0; j < n; j ++) {
			//if(j == i) continue;
			//printf("t = %0.6f, %0.6f, %0.6f; %0.6f, %0.6f, %0.6f; %0.6f, %0.6f, %0.6f \n", tx[j], ty[j], tz[j], tx[i], ty[i], tz[i], fabs( tx[j]-tx[i] ), fabs( ty[j]-ty[i] ), fabs( tz[j]-tz[i] ));
			if( fabs( tx[j]-tx[i] ) < thresh &&
				fabs( ty[j]-ty[i] ) < thresh && 
				fabs( tz[j]-tz[i] ) < thresh) {
					inlierNum ++;
					scale[0] += tx[j]; scale[1] += ty[j]; scale[2] += tz[j];
			}
		}

		if( inlierNum > inlierNumMax ) {
			inlierNumMax = inlierNum;
			t[0] = scale[0] / inlierNum; t[1] = scale[1] / inlierNum; t[2] = scale[2] / inlierNum;
		}
		if( (float)inlierNumMax > 0.8*(float)n) {
			break;
		}
	}
	//printf("inlierNum = %d / %d, ", inlierNumMax, n);
	return;

}

void compute_t_scale(int n, v3_t *p1_3d, v3_t *p2_3d, double *R, double *t, int *inlierList) {
	double tmp[9], tmp2[3], Qv[3], PQ_tv[3], tt[3];
	double scale[3], prob[3];
	double scale_t = 0, maxScale = 0, sumAbs = 0, sumt = 0;
	int cntPoint = 0;
	double *tx, *ty, *tz;

	v3_t Q, PQ, PQ_t;
	int i,  maxid = 0;

	tx = (double *)malloc(sizeof(double) * n);
	ty = (double *)malloc(sizeof(double) * n);
	tz = (double *)malloc(sizeof(double) * n);

	for(i = 0; i < 3; i ++) scale[i] = 0;
	for (i = 0; i < n; i++) {
		if( inlierList[i] < 1 ) continue;
		Q = p1_3d[i]; PQ_t = p2_3d[i];
		if( -Vz(Q) < 0.01 || -Vz(Q) > 3.0 || -Vz(PQ_t) < 0.01 || -Vz(PQ_t) > 3.0 ) continue;
		Qv[0] = Vx(Q), Qv[1] = Vy(Q), Qv[2] = Vz(Q);
		PQ_tv[0] = Vx(PQ_t), PQ_tv[1] = Vy(PQ_t), PQ_tv[2] = Vz(PQ_t);

		matrix_product331(R, Qv, tmp); // tmp = R * Q
		
		cntPoint ++;
		tt[0] = PQ_tv[0] - tmp[0]; tt[1] = PQ_tv[1] - tmp[1]; tt[2] = PQ_tv[2] - tmp[2];
		scale[0] += tt[0]; scale[1] += tt[1]; scale[2] += tt[2]; 

		tx[cntPoint-1] = tt[0]; ty[cntPoint-1] = tt[1]; tz[cntPoint-1] = tt[2];
	}

	if(cntPoint < 1) return;

	compute_t_ransac(cntPoint, tx, ty, tz, t);
	//printf("t = %0.6f %0.6f %0.6f / %0.6f %0.6f %0.6f \n", t[0], t[1], t[2], scale[0]/cntPoint, scale[1]/cntPoint, scale[2]/cntPoint);
	
	// average
	//for(i = 0; i < 3; i ++) {
	//	scale[i] = scale[i] / cntPoint;
	//	t[i] = scale[i]; 
	//}
	free(tx); free(ty); free(tz);
}

int compute_pose_ransac(int n, v2_t *r_pts, v2_t *l_pts, v3_t *r_pts_3d,  v3_t *l_pts_3d,
						double *K1, double *K2, 
						double ransac_threshold, int ransac_rounds, 
						double *R_out, double *t_out, int *inlierList)
{
	v2_t *r_pts_norm, *l_pts_norm;
	int i, j, round;
	double thresh_norm;
	double K1_inv[9], K2_inv[9];
	int max_inliers = 0;
	double min_score = DBL_MAX;
	double E_best[9];
	v2_t r_best, l_best;
	int *inlierListTmp;
	//in opencv
	//float matrix[9] = { 0 };
	//int *points1, *points2;
	//if (n >= 8){
	//	points1 = (int*)malloc(sizeof(int)*n * 2);
	//	points2 = (int*)malloc(sizeof(int)*n * 2);
	//	for (i = 0; i < n * 2; i += 2){
	//		points1[i] = Vx(r_pts[i]);
	//		points1[i + 1] = Vy(r_pts[i]);
	//		points2[i] = Vx(l_pts[i]);
	//		points2[i + 1] = Vy(l_pts[i]);
	//	}
	//	cvFindFundamentalMatrix(points2, points1, n, CV_FM_RANSAC, matrix);

	//	free(points1);
	//	free(points2);
	//}

	inlierListTmp = (int *)malloc(sizeof(int) * n);
	r_pts_norm = (v2_t *)malloc(sizeof(v2_t) * n);
	l_pts_norm = (v2_t *)malloc(sizeof(v2_t) * n);

	matrix_invert(3, K1, K1_inv);
	matrix_invert(3, K2, K2_inv);

	for (i = 0; i < n; i++) {
		double r[3] = { Vx(r_pts[i]), Vy(r_pts[i]), 1.0 };
		double l[3] = { Vx(l_pts[i]), Vy(l_pts[i]), 1.0 };

		double r_norm[3], l_norm[3];

		matrix_product331(K1_inv, r, r_norm);
		matrix_product331(K2_inv, l, l_norm);

		r_pts_norm[i] = v2_new(-r_norm[0], -r_norm[1]);
		l_pts_norm[i] = v2_new(-l_norm[0], -l_norm[1]);

	}

	thresh_norm = ransac_threshold * ransac_threshold;

	for (round = 0; round < ransac_rounds; round++) {
		/* Pick 5 random points */
		v2_t r_pts_inner[5], l_pts_inner[5];
		int indices[5];
		int num_hyp;
		double E[90];
		int inliers_hyp[10];
		int first_hyp = -1, first_hyp_idx = -1, second_hyp = -1;
		int best = 0;
		int num_ident = 0;
		int inliers = 0;


		choose(n, 5, indices);

		for (i = 0; i < 5; i++) {
			r_pts_inner[i] = r_pts_norm[indices[i]];
			l_pts_inner[i] = l_pts_norm[indices[i]];

			/* Check for degeneracy */
			if (Vx(r_pts_inner[i]) == Vx(l_pts_inner[i]) &&
				Vy(r_pts_inner[i]) == Vy(l_pts_inner[i]))
				num_ident++;
		}

		if (num_ident >= 3)
			continue;  /* choose another 5 */

		generate_Ematrix_hypotheses(5, r_pts_inner, l_pts_inner, &num_hyp, E);

		for (i = 0; i < num_hyp; i++) {
			int best_inlier;
			double score = 0.0;

			double E2[9], tmp[9], F[9];
			memcpy(E2, E + 9 * i, 9 * sizeof(double));
			E2[0] = -E2[0];
			E2[1] = -E2[1];
			E2[3] = -E2[3];
			E2[4] = -E2[4];
			E2[8] = -E2[8];

			matrix_transpose_product(3, 3, 3, 3, K2_inv, E2, tmp);
			matrix_product(3, 3, 3, 3, tmp, K1_inv, F);

			//if (n >= 8){
			//	for (int i = 0; i < 9; i++)
			//		F[i] = matrix[i];
			//	matrix_transpose_product(3, 3, 3, 3, K2, F, tmp);
			//	matrix_product(3, 3, 3, 3, tmp, K1, E2);
			//	E2[0] = -E2[0];
			//	E2[1] = -E2[1];
			//	E2[3] = -E2[3];
			//	E2[4] = -E2[4];
			//	E2[8] = -E2[8];
			//	memcpy(E + 9 * i, E2, 9 * sizeof(double));
			//}

			for(j = 0; j < n; j ++) inlierListTmp[j] = 0;
			inliers = evaluate_Ematrix(n, r_pts, l_pts, // r_pts_norm, l_pts_norm, 
				thresh_norm, F, // E + 9 * i, 
				&best_inlier, &score, inlierListTmp);

			if (inliers > max_inliers ||
				(inliers == max_inliers && score < min_score)) {
					best = 1;
					max_inliers = inliers;
					min_score = score;
					memcpy(E_best, E + 9 * i, sizeof(double) * 9);
					r_best = r_pts_norm[best_inlier];
					l_best = l_pts_norm[best_inlier];
					memcpy(inlierList, inlierListTmp, sizeof(int)*n);

					ransac_rounds = (abs(log(1 - 0.99) / log(1 - pow((double)inliers / n, 5))) > ransac_rounds) ?
					ransac_rounds : abs(log(1 - 0.99) / log(1 - pow((double)inliers / n, 5)));
					
			}

			inliers_hyp[i] = inliers;
		}

		if (best) {
			for (i = 0; i < num_hyp; i++) {
				if (inliers_hyp[i] > first_hyp) {
					first_hyp = inliers_hyp[i];
					first_hyp_idx = i;
				}
			}

			for (i = 0; i < num_hyp; i++) {
				if (i != first_hyp_idx && inliers_hyp[i] > second_hyp) {
					second_hyp = inliers_hyp[i];
				}
			}

			// printf("first: %d, second: %d\n", first_hyp, second_hyp);
		}
	}

	free(inlierListTmp);

	if (max_inliers > 0) {
		int best_inlier;
		double score;

		int success = 
			find_extrinsics_essential_multipt(E_best, n, 
			r_pts_norm, l_pts_norm,
			R_out, t_out);
		//int success = 
		//	find_extrinsics_essential_multipt_depth_support(E_best, n, 
		//	r_pts_norm, l_pts_norm, r_pts_3d, l_pts_3d, 
		//	R_out, t_out, inlierList);

		if (success == 0) {
			free(r_pts_norm);
			free(l_pts_norm);
			return 0;
		}

		compute_t_scale(n, r_pts_3d, l_pts_3d, R_out, t_out, inlierList);
	}

	free(r_pts_norm);
	free(l_pts_norm);

	return max_inliers;
}

