#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <thrust/device_vector.h>

#include "qbmatmul.h"
#include "common/cucheck.h"
#include "common/math/matgen.h"
#include "common/math/numerr.h"
#include "common/debug/cuprint.h"


int main() {
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;
    curandGenerator_t curandGen;

    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverHandle));
    CURAND_CHECK(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curandGen, 1234ULL));

    const int m = 128;
    const int n = 96;
    const int true_rank = 4;
    const int max_rank = 8;
    const double eps = 1e-8;

    std::vector<double> s_values = {10.0, 5.0, 1.0, 0.1};

    thrust::device_vector<double> d_A = generate_low_rank_matrix(
        m, n, true_rank, s_values,
        cublasHandle, cusolverHandle, curandGen);
    thrust::device_vector<double> d_A_orig = d_A;

    // debug
    fprint_device_array2d("build/dA.txt", d_A.data().get(), m * sizeof(double), m * sizeof(double), n);

    thrust::device_vector<double> d_Q_final((size_t)m * max_rank);
    thrust::device_vector<double> d_B_final((size_t)max_rank * n);
    std::vector<int> keep_cols;

    int actual_rank = qb_factorization(
        d_A.data().get(),
        m,
        n,
        max_rank,
        eps,
        d_Q_final.data().get(),
        d_B_final.data().get(),
        keep_cols,
        cublasHandle,
        cusolverHandle);

    std::cout << "QB factorization finished with rank = " << actual_rank << "\n";
    if (actual_rank <= 0) {
        std::cerr << "QB factorization returned nonpositive rank.\n";
        return EXIT_FAILURE;
    }


    thrust::device_vector<double> d_recon((size_t)m * n);
    const double one = 1.0;
    const double zero = 0.0;

    CUBLAS_CHECK(cublasDgemm(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        actual_rank,
        &one,
        d_Q_final.data().get(),
        m,                     // lda
        d_B_final.data().get(),
        max_rank,              // ldb (分配的行数)
        &zero,
        d_recon.data().get(),
        m));                   // ldc

    CUDA_CHECK(cudaDeviceSynchronize());

    // debug
    fprint_device_array2d("build/dRecon.txt", d_recon.data().get(), m * sizeof(double), m * sizeof(double), n);

    Error<double> err = norm2_error<double>(
        cublasHandle,
        d_A_orig.data().get(),
        d_recon.data().get(),
        m,
        n);

    std::cout << err << "\n";

    CURAND_CHECK(curandDestroyGenerator(curandGen));
    CUBLAS_CHECK(cublasDestroy(cublasHandle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverHandle));

    return EXIT_SUCCESS;
}

    