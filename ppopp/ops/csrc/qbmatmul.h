#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <vector>
#include <cmath>
#include <cstdio>


#include "common/cucheck.h"
#include "kernel.h"


/**
 * @brief QB Factorization
 * 
 * @param d_A_ptr               Input matrix
 * @param m 
 * @param n 
 * @param k 
 * @param eps                   Relative significance ratio
 * @param d_Q_final_ptr 
 * @param d_B_final_ptr 
 * @param h_keep_cols_global    Index of remaining cols
 * @param cublasHandle
 * @param cusolverHandle
 * @return int                  Final rank
 */
int qb_factorization(
    double* d_A_ptr,
    int m,
    int n,
    int k,
    double eps,
    double* d_Q_final_ptr,
    double* d_B_final_ptr,
    std::vector<int>& h_keep_cols_global,
    cublasHandle_t cublasHandle,
    cusolverDnHandle_t cusolverHandle
);

