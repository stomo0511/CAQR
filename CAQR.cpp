//
//  CAQR.cpp
//
//  Created by T. Suzuki on 2016/06/08.
//  Copyright (c) 2013 T. Suzuki. All rights reserved.
//

#include <iostream>
#include <omp.h>
#include <cassert>
#include <cstdlib>
#include <cmath>

#include <lapacke.h>

#include <CoreBlasTile.hpp>
#include <TMatrix.hpp>

using namespace std;

void Check_Accuracy( const int M, const int N, double *A, double *Q, double *R );

/**
 * ToDo:
 * ・OpenMP並列化
 * ・OpenMP task depend
 * ・Processing アニメーション
 */

int main(int argc, const char * argv[])
{
	if (argc < 6)
	{
		cerr << "Usage: a.out [M] [N] [P] [NB] [IB]\n";
		exit (1);
	}
	
	const int M =  atoi(argv[1]);  // # rows of the matrix
	const int N =  atoi(argv[2]);  // # columns of the matrix
	const int P =  atoi(argv[3]);  // # domains
	const int NB = atoi(argv[4]);  // size of square tile
	const int IB = atoi(argv[5]);  // inner blocking size

	// Check command line arguments
	assert( M >= N );
	assert( NB >= IB );
	assert( N >= NB );

	//////////////////////////////////////////////////////////////////////
	// Definitions and Initialize
	TMatrix A(M,N,NB,NB,IB);
	A.Set_Rnd( 20160620 );
	
	const int MT = A.mt();
	const int NT = A.nt();
	
	const int MTl = (MT % P) ==0 ? MT/P : MT/P + 1;

	assert( MTl >= NT );

	#ifdef DEBUG
	cout << "KN,I1,I2,J,K,Time" << endl;
	#endif

	// refered in workspace.c of PLASMA
	TMatrix T0(MT*IB,NT*NB,IB,NB,IB);
	TMatrix T1((P-1)*IB,NT*NB,IB,NB,IB);
	
	#ifdef DEBUG
	// Copy the elements of TMatrix class A to double array mA
	double *mA = new double [ M*N ];
	A.Array_Copy(mA);
	#endif

	// Definitions and Initialize　END
	//////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////
	// Timer start
	double time = omp_get_wtime();

	//////////////////////////////////////////////////////////////////////
	// Semi-Parallel Tile CAQR
	//
	int nextMT = MTl;
	int proot = 0;

	for (int k=0; k<NT; k++)
	{
		if ( k > nextMT )
		{
			proot++;
			nextMT += MTl;
		}

		//////////////////////////////////////////////////////////
		// PLASMA-like factorization in each domain
		for (int p=proot; p<P; p++)
		{
			int ibeg = 0;
			if ( p == proot )
			{
				ibeg = k - proot*MTl;
			}
			GEQRT( A(p*MTl+ibeg,k), T0(p*MTl+ibeg,k) );
			#ifdef DEBUG
			cout << "1," << p*MTl+ibeg << ",0," << k << "," << k << "," << omp_get_wtime() - time << endl;
			#endif

			for (int j=k+1; j<NT; j++)
			{
				LARFB( PlasmaLeft, PlasmaTrans,
						A(p*MTl+ibeg,k), T0(p*MTl+ibeg,k), A(p*MTl+ibeg,j) );
				#ifdef DEBUG
				cout << "2," << p*MTl+ibeg << ",0," << j << "," << k << "," << omp_get_wtime() - time << endl;
				#endif
			}
			for (int i=ibeg+1; (i<MTl) && (p*MTl+i<MT); i++)
			{
				TSQRT( A(p*MTl+ibeg,k), A(p*MTl+i,k), T0(p*MTl+i,k) );
				#ifdef DEBUG
				cout << "3," << p*MTl+i << ",0," << k << "," << k << "," << omp_get_wtime() - time << endl;
				#endif

				for (int j=k+1; j<NT; j++)
				{
					SSRFB( PlasmaLeft, PlasmaTrans,
							A(p*MTl+i,k), T0(p*MTl+i,k), A(p*MTl+ibeg,j), A(p*MTl+i,j) );
					#ifdef DEBUG
					cout << "4," << p*MTl+i << ",0," << j << "," << k << "," << omp_get_wtime() - time << endl;
					#endif
				}
			}
		}

		//////////////////////////////////////////////////////////
		// Merge
		for (int m=1; m<=(int)ceil(log2(P - proot)); m++)
		{
			int p1 = proot;
			int p2 = p1 + (int)pow(2,m-1);
			while (p2 < P)
			{
				int i1 = 0;
				int i2 = 0;
				if ( p1 == proot )
				{
					i1 = k - proot*MTl;
				}

				TTQRT( A(p1*MTl+i1,k), A(p2*MTl+i2,k), T1(p2-1,k) );
				#ifdef DEBUG
				cout << "5," << p1*MTl+i1 << "," << p2*MTl+i2 << "," << k << "," << k << "," << omp_get_wtime() - time << endl;
				#endif

				for (int j=k+1; j<NT; j++)
				{
					TTMQR( PlasmaLeft, PlasmaTrans,
							A(p2*MTl+i2,k), T1(p2-1,k), A(p1*MTl+i1,j), A(p2*MTl+i2,j) );
					#ifdef DEBUG
					cout << "6," << p1*MTl+i1 << "," << p2*MTl+i2 << "," << j << "," << k << "," << omp_get_wtime() - time << endl;
					#endif
				}
				p1 += (int)pow(2,m);
				p2 += (int)pow(2,m);
			}
		}
	}
	// Semi-Parallel Tile CAQR END
	//////////////////////////////////////////////////////////////////////
	
	return EXIT_SUCCESS;
}

void Check_Accuracy( const int M, const int N, double *mA, double *mQ, double *mR )
{
  ////////////////////////////////////////////////////////////////////////////
  // Check Orthogonarity

  // Set Id to the identity matrix
  int mn = std::min(M,N);
  double* Id = new double[ mn * mn ];
  for (int i=0; i<mn; i++)
    for (int j=0; j<mn; j++)
      Id[ i + j*mn ] = (i == j) ? 1.0 : 0.0;

  double alpha = 1.0;
  double beta  = -1.0;

  cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
	      N, M, alpha, mQ, M, beta, Id, N);

  ////////////////////////////////////////////////////////////////////////////
  double* Work = new double[ mn ];
  double normQ = LAPACKE_dlansy_work(LAPACK_COL_MAJOR, 'F', 'U',
				     mn, Id, mn, Work);

  std::cout << "norm(I-Q*Q') = " << normQ << std::endl;

  delete [] Work;
  delete [] Id;
  // Check Orthogonarity END
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Check Residure
  double* QR = new double[ M * N ];
  alpha = 1.0;
  beta  = 0.0;

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	      M, N, M, alpha, mQ, M, mR, M, beta, QR, M);

  /////////////////////////////////////////////////////////////////////////////
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      QR[ i + j*M ] -= mA[ i + j*M ];

  Work = new double[ M ];
  normQ = LAPACKE_dlange_work(LAPACK_COL_MAJOR, 'F',
			      M, N, QR, M, Work);
  std::cout << "norm(A-Q*R) = " << normQ << std::endl;

  delete [] Work;
  delete [] QR;
  // Check Residure END
  ////////////////////////////////////////////////////////////////////////////
}

