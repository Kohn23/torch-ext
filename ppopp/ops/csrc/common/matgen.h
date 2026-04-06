#pragma once

#include <vector>
#include <cstddef>
#include <type_traits>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <thrust/device_vector.h>

typedef enum {
    GEOMETRIC,
    GEOMETRIC_ZERO,
    UNIFORM,
    CLUSTER0,
    CLUSTER1,
    ARITHMETIC,
    NORMAL
}DistributionType;

/**
 * @brief Generate a normal (Gaussian) random matrix.
 * 
 * @param gen    cuRAND generator.
 * @param d_vec  Output device vector.
 * @param m      Number of rows.
 * @param n      Number of columns.
 */
template <typename T>
void generateNormalMatrix(curandGenerator_t& gen, thrust::device_vector<T>& d_vec, size_t m, size_t n)
{
    T* raw_ptr = thrust::raw_pointer_cast(d_vec.data());

    if constexpr (std::is_same_v<T, float>) {
        curandGenerateNormal(gen, raw_ptr, m * n, 0.0f, 1.0f);
    } else if constexpr (std::is_same_v<T, double>) {
        curandGenerateNormalDouble(gen, raw_ptr, m * n, 0.0, 1.0);
    }
}

/**
 * @brief CUDA kernel to scale columns of a matrix by a vector of scaling factors.
 */
template <typename T>
__global__ void scale_columns_kernel(
    T* M_out,
    const T* M_in,
    const T* vec,
    size_t rows,
    size_t cols,
    size_t ld_out,
    size_t ld_in)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        T scale_factor = vec[col];
        M_out[col * ld_out + row] = M_in[col * ld_in + row] * scale_factor;
    }
}


/**
 * @brief Generate singular values according to a specified distribution.
 * 
 * @param type       Distribution type
 * @param max_val    Largest singular value.
 * @param min_val    Smallest singular value (for distributions that use it).
 * @param s_values   Output vector of singular values (size rank).
 * @param rank       Number of singular values to generate.
 */
void generate_singular_values(
    DistributionType type,
    double max_val,
    double min_val,
    std::vector<double>& s_values,
    size_t rank
);

/**
 * @brief Generate a random orthogonal matrix of size rows x cols (rows >= cols).
 * 
 * The matrix is obtained by generating a Gaussian random matrix (rows x cols),
 * then performing a QR decomposition and taking the Q factor.
 * 
 * @param d_Q             Output device pointer (rows * cols, column-major).
 * @param rows            Number of rows.
 * @param cols            Number of columns (rank).
 * @param cusolverHandle  cuSOLVER handle.
 * @param curandGen       cuRAND generator.
 * @param seed            Seed for random numbers.
 */
void generate_random_orthogonal_matrix(
    double* d_Q, int rows, int cols,
    cusolverDnHandle_t cusolverHandle,
    curandGenerator_t curandGen,
    unsigned long long seed);

/**
 * @brief Generate a random low-rank matrix on the GPU.
 * 
 * A = Q1 * diag(S) * Q2^T, where Q1 (m x rank) and Q2 (n x rank) are random
 * orthogonal matrices (obtained via QR of Gaussian random matrices), and S
 * is a diagonal matrix with given singular values.
 * 
 * @param m               Number of rows.
 * @param n               Number of columns.
 * @param rank            Rank of the matrix (size of S).
 * @param s_values        Vector of singular values (length rank).
 * @param cublasHandle    cuBLAS handle.
 * @param cusolverHandle  cuSOLVER handle.
 * @param curandGen       cuRAND generator (already created).
 * @param seed1           Seed for generating Q1.
 * @param seed2           Seed for generating Q2.
 * @return thrust::device_vector<double>  Matrix A in column-major order (size m*n).
 */
thrust::device_vector<double> generate_low_rank_matrix(
    int m, int n, size_t rank,
    const std::vector<double>& s_values,
    cublasHandle_t cublasHandle,
    cusolverDnHandle_t cusolverHandle,
    curandGenerator_t curandGen,
    unsigned long long seed1 = 1234ULL,
    unsigned long long seed2 = 5678ULL
);
