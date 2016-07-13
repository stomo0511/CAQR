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
	cout << "Size of matrix: M = " << M << ", N = " << N << endl;
	cout << "Size of square tile: NB = " << NB << endl;
	cout << "Width of inner blocks: IB = " << IB << endl;
	cout << "Number of tiles: MT = " << MT << ", NT = " << NT << endl;
	cout << "Number of domains: P = " << P << endl;
	cout << "Number of tile rows within a domain: MTL = " << MTl << endl;
	cout << "clock precision = " << omp_get_wtick() << endl;
	#endif

	// refered in workspace.c of PLASMA
	TMatrix T0(MT*IB,NT*NB,IB,NB,IB);
	TMatrix T1((P-1)*IB,NT*NB,IB,NB,IB);
	
	#ifdef DEBUG
	// Copy the elements of TMatrix class A to double array mA
	double *mA = new double [ M*N ];
	A.Array_Copy(mA);
	#endif

	//////////////////////////////////////////////////////////////////////
	// List Item
	int **Ap, **Am;

	Ap = (int **)malloc(sizeof(int *) * MT);
	for (int i=0; i<MT; i++)
		Ap[i] = (int *)malloc(sizeof(int) * NT);

	Am = (int **)malloc(sizeof(int *) * (P-1));
		for (int i=0; i<(P-1); i++)
			Am[i] = (int *)malloc(sizeof(int) * NT);
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

	#pragma omp parallel firstprivate(time)
	{
		#pragma omp single
		{
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

					#pragma omp task depend(inout:Ap[p*MTl+ibeg][k])
					{
						GEQRT( A(p*MTl+ibeg,k), T0(p*MTl+ibeg,k) );
						#ifdef DEBUG
						#pragma omp critical
//						cout << "GEQRT(" << k << "," << p*MTl+ibeg << "," << k << ") : " << omp_get_thread_num() << " : " << omp_get_wtime() - time << endl;
						cout << "1," << p*MTl+ibeg << ",0," << k << "," << k << "," << omp_get_wtime() - time << endl;
						#endif
					}

					for (int j=k+1; j<NT; j++)
					{
						#pragma omp task depend(in:Ap[p*MTl+ibeg][k]) depend(inout:Ap[p*MTl+ibeg][j])
						{
							LARFB( PlasmaLeft, PlasmaTrans,
									A(p*MTl+ibeg,k), T0(p*MTl+ibeg,k), A(p*MTl+ibeg,j) );
							#ifdef DEBUG
							#pragma omp critical
//							cout << "LARFB(" << k << "," << p*MTl+ibeg << "," << j << ") : " << omp_get_thread_num() << " : " << omp_get_wtime() - time << endl;
							cout << "2," << p*MTl+ibeg << ",0," << j << "," << k << "," << omp_get_wtime() - time << endl;
							#endif
						}
					}
					// End of j-loop

					for (int i=ibeg+1; (i<MTl) && (p*MTl+i<MT); i++)
					{
						#pragma omp task depend(inout:Ap[p*MTl+ibeg][k]) depend(out:Ap[p*MTl+i][k])
						{
							TSQRT( A(p*MTl+ibeg,k), A(p*MTl+i,k), T0(p*MTl+i,k) );
							#ifdef DEBUG
							#pragma omp critical
//							cout << "TSQRT(" << k << "," << p*MTl+i << "," << k << ") : " << omp_get_thread_num() << " : " << omp_get_wtime() - time << endl;
							cout << "3," << p*MTl+i << ",0," << k << "," << k << "," << omp_get_wtime() - time << endl;
							#endif
						}

						for (int j=k+1; j<NT; j++)
						{
							#pragma omp task depend(in:Ap[p*MTl+i][k]) depend(inout:Ap[p*MTl+ibeg][j],Ap[p*MTl+i][j])
							{
								SSRFB( PlasmaLeft, PlasmaTrans,
										A(p*MTl+i,k), T0(p*MTl+i,k), A(p*MTl+ibeg,j), A(p*MTl+i,j) );
								#ifdef DEBUG
								#pragma omp critical
								cout << "SSRFB(" << k << "," << p*MTl+i << "," << j << ") : " << omp_get_thread_num() << " : " << omp_get_wtime() - time << endl;
								#endif
							}
						}
						// End of j-loop
					}
					// End of i-loop
				}
				// End of p-loop

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

						#pragma omp task depend(inout:Ap[p1*MTl+i1][k],Ap[p2*MTl+i2][k]) depend(out:Am[p2-1][k])
						{
							TTQRT( A(p1*MTl+i1,k), A(p2*MTl+i2,k), T1(p2-1,k) );
							#ifdef DEBUG
							#pragma omp critical
							cout << "TTQRT(" << k << "," << p1*MTl+i1 << "," << p2*MTl+i2 << "," << k << ") : "  << omp_get_thread_num() << " : " << omp_get_wtime() - time << endl;
							#endif
						}

						for (int j=k+1; j<NT; j++)
						{
							#pragma omp task depend(in:Ap[p2*MTl+i2][k],Am[p2-1][k]) depend(inout:Ap[p1*MTl+i1][j],Ap[p2*MTl+i2][j])
							{
								TTMQR( PlasmaLeft, PlasmaTrans,
										A(p2*MTl+i2,k), T1(p2-1,k), A(p1*MTl+i1,j), A(p2*MTl+i2,j) );
								#ifdef DEBUG
								#pragma omp critical
								cout << "TTMQR(" << k << "," << p1*MTl+i1 << "," << p2*MTl+i2 << "," << j << ") : "  << omp_get_thread_num() << " : " << omp_get_wtime() - time << endl;
								#endif
							}
						}
						p1 += (int)pow(2,m);
						p2 += (int)pow(2,m);
					}
					// End of while-loop
				}
				// End of m-loop
			}
			// End of k-loop
		}
		// End of omp single region
	}
	// End of omp parallel region
	// Semi-Parallel Tile CAQR END
	//////////////////////////////////////////////////////////////////////
	
	// Timer stop
	time = omp_get_wtime() - time;
	cout << M << ", " << N << ", " << NB << ", " << IB << ", " << time << endl;

	for (int i=0; i<MT; i++)
		free(Ap[i]);
	free(Ap);
	for (int i=0; i<(P-1); i++)
		free(Am[i]);
	free(Am);
	/////////////////////////////////////////////////////////////////////////////////////////////////
	#ifdef DEBUG
	cout << "\n Check Accuracy start.\n";

	//////////////////////////////////////////////////////////////////////
	// Regenerate Q
	TMatrix Q(M,M,NB,NB,IB);

	// Set to the identity matrix
	Q.Set_Iden();

	//////////////////////////////////////////////////////////////////////
	// Make Orthogonal matrix Q
	//
	// Inverse order of Semi-Parallel Tile CAQR
	nextMT = MTl;
	proot = 0;

	for (int k=NT-1; k>=0; k--)
	{
		if ( k > nextMT )
		{
			proot++;
			nextMT += MTl;
		}
		// If MTl >= NT then proot = 0.

		//////////////////////////////////////////////////////////
		// Merge
		for (int m=(int)ceil(log2(P - proot)); m>0; m--)
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
				#pragma omp parallel for
				for (int j=k; j<Q.nt(); j++)
				{
					TTMQR( PlasmaLeft, PlasmaNoTrans,
							A(p2*MTl+i2,k), T1(p2-1,k), Q(p1*MTl+i1,j), Q(p2*MTl+i2,j) );
//					#ifdef DEBUG
//					cout << "TTMQR(" << k << "," << p1*MTl+i1 << "," << p2*MTl+i2 << "," << j << ")" << endl;
//					#endif
				}
				p1 += (int)pow(2,m);
				p2 += (int)pow(2,m);
			} // END of while
		} // END of m-loop

		//////////////////////////////////////////////////////////
		// In each domain
		for (int p=proot; p<P; p++)
		{
			int ibeg = 0;
			if ( p == proot )
			{
				ibeg = k - proot*MTl;
			}
			for (int i = (p+1)*MTl > MT ? (MT-p*MTl)-1 : MTl-1; i>ibeg; i--)
			{
				#pragma omp parallel for
				for (int j=k; j<Q.nt(); j++)
				{
					SSRFB( PlasmaLeft, PlasmaNoTrans,
							A(p*MTl+i,k), T0(p*MTl+i,k), Q(p*MTl+ibeg,j), Q(p*MTl+i,j) );
//					#ifdef DEBUG
//					cout << "SSRFB(" << k << "," << p*MTl+i << "," << j << ")" << endl;
//					#endif
				}
			}
			#pragma omp parallel for
			for (int j=k; j<Q.nt(); j++)
			{
				LARFB( PlasmaLeft, PlasmaNoTrans,
						A(p*MTl+ibeg,k), T0(p*MTl+ibeg,k), Q(p*MTl+ibeg,j) );
//				#ifdef DEBUG
//				cout << "LARFB(" << k << "," << p*MTl+ibeg << "," << j << ")" << endl;
//				#endif
			}
		} // END of p-loop
	} // END of k-loop
	// Inverse order of Semi-Parallel Tile CAQR END
	// Regenerate Q END
	//////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////
	// Check Accuracy
	double *mQ = new double [ M*M ];
	double *mR = new double [ M*N ];

	Q.Array_Copy(mQ);
	A.Array_Copy(mR);

	for (int i=0; i<M; i++)
		for (int j=0; j<N; j++)
			if (i > j)
				mR[ i + M*j ] = 0.0;

	Check_Accuracy( M, N, mA, mQ, mR );
	// Check Accuracy END
	//////////////////////////////////////////////////////////////////////

	delete [] mA;
	delete [] mQ;
	delete [] mR;

	cout << "Done\n";
	#endif

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

