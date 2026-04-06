#include "common/matgen.h"

#include <cmath>
#include <cstdio>
#include <algorithm>



void generate_singular_values(
    DistributionType type,
    double max_val,
    double min_val,
    std::vector<double>& s_values,
    size_t rank)
{
    s_values.resize(rank);
    if (rank == 0) return;

    if (type == GEOMETRIC) {
        printf("Generating GEOMETRICALLY distributed singular values...\n");
        if (rank == 1) {
            s_values[0] = max_val;
        } else {
            double log_max = log(max_val);
            double log_min = log(min_val);
            double step = (log_min - log_max) / (double)(rank - 1);
            for (size_t i = 0; i < rank; ++i) {
                s_values[i] = exp(log_max + (double)i * step);
            }
        }
    }
    else if (type == GEOMETRIC_ZERO) {
        printf("Generating GEOMETRICALLY distributed singular values with an internal ZERO block...\n");
        if (rank == 1) {
            s_values[0] = max_val;
        } else {
            size_t zero_count = std::max<size_t>(1, rank / 6);
            size_t zero_start = (rank > zero_count) ? (rank - zero_count) / 2 : 0;
            size_t zero_end = std::min(rank, zero_start + zero_count);
            size_t nonzero_count = rank - (zero_end - zero_start);

            double min_positive = (min_val > 0.0) ? min_val : max_val * 1e-6;
            double log_max = log(max_val);
            double log_min = log(min_positive);
            double step = (nonzero_count > 1) ? (log_min - log_max) / (double)(nonzero_count - 1) : 0.0;

            size_t nz_pos = 0;
            for (size_t i = 0; i < rank; ++i) {
                if (i >= zero_start && i < zero_end) {
                    s_values[i] = 0.0;
                } else {
                    double val = (nonzero_count == 1) ? max_val : exp(log_max + (double)nz_pos * step);
                    s_values[i] = val;
                    ++nz_pos;
                }
            }
        }
    }
    else if (type == UNIFORM) {
        printf("Generating UNIFORMLY (linearly) distributed singular values...\n");
        if (rank == 1) {
            s_values[0] = max_val;
        } else {
            double step = (max_val - min_val) / (double)(rank - 1);
            for (size_t i = 0; i < rank; ++i) {
                s_values[i] = max_val - (double)i * step;
            }
        }
    }
    else if (type == CLUSTER0) {
        printf("Generating 'Cluster0' singular values (sharp drop)...\n");
        size_t cutoff_rank = rank / 4;
        if (cutoff_rank == 0 && rank > 0) cutoff_rank = 1;
        
        double high_end_val = max_val * 0.9;
        double step = (cutoff_rank > 1) ? (max_val - high_end_val) / (double)(cutoff_rank - 1) : 0.0;
        
        for (size_t i = 0; i < rank; ++i) {
            if (i < cutoff_rank) {
                s_values[i] = max_val - (double)i * step;
            } else {
                s_values[i] = min_val;
            }
        }
    }
    else if (type == CLUSTER1) {
        printf("Generating 'Cluster1' singular values (staircase)...\n");
        size_t cutoff1 = rank / 3;
        size_t cutoff2 = 2 * rank / 3;
        double mid_val = (max_val + min_val) / 2.0;

        for (size_t i = 0; i < rank; ++i) {
            if (i < cutoff1) {
                s_values[i] = max_val;
            } else if (i < cutoff2) {
                s_values[i] = mid_val;
            } else {
                s_values[i] = min_val;
            }
        }
    }
    else if (type == ARITHMETIC) {
        printf("Generating ARITHMETIC progression singular values...\n");
        if (rank == 1) {
            s_values[0] = max_val;
        } else {
            double step = (max_val - min_val) / (double)(rank - 1);
            for (size_t i = 0; i < rank; ++i) {
                s_values[i] = max_val - (double)i * step;
            }
        }
    }
    else if (type == NORMAL) {
        printf("Generating NORMAL (Gaussian-like) distributed singular values...\n");
        double mean = (double)rank / 2.0;
        double sigma = (double)rank / 6.0; 
        
        for (size_t i = 0; i < rank; ++i) {
            double x = (double)i;
            double gaussian_weight = exp(-0.5 * pow((x - mean) / sigma, 2.0));
            s_values[i] = min_val + (max_val - min_val) * gaussian_weight;
        }
    }
    else {
        fprintf(stderr, "Error: Unknown spectrum type '%s'.\n", type.c_str());
        exit(1);
    }
}


void generate_random_orthogonal_matrix(
    double* d_Q, int rows, int cols,
    cusolverDnHandle_t cusolverHandle,
    curandGenerator_t curandGen,
    unsigned long long seed)
{
    // Set seed
    curandSetPseudoRandomGeneratorSeed(curandGen, seed);
    
    // Generate random Gaussian matrix (rows x cols)
    generateNormalMatrix(curandGen, thrust::device_vector<double>(d_Q, d_Q + rows * cols), rows, cols);

    // Compute QR decomposition
    int lwork_geqrf = 0, lwork_orgqr = 0;
    cusolverDnDgeqrf_bufferSize(cusolverHandle, rows, cols, d_Q, rows, &lwork_geqrf);
    cusolverDnDorgqr_bufferSize(cusolverHandle, rows, cols, cols, d_Q, rows, d_Q, &lwork_orgqr);
    int lwork = std::max(lwork_geqrf, lwork_orgqr);
    
    thrust::device_vector<double> work(lwork);
    thrust::device_vector<double> tau(cols);
    thrust::device_vector<int> devInfo(1);
    
    cusolverDnDgeqrf(cusolverHandle, rows, cols, d_Q, rows,
                     tau.data().get(), work.data().get(), lwork,
                     devInfo.data().get());
    
    cusolverDnDorgqr(cusolverHandle, rows, cols, cols, d_Q, rows,
                     tau.data().get(), work.data().get(), lwork,
                     devInfo.data().get());
}


thrust::device_vector<double> generate_low_rank_matrix(
    int m, int n, size_t rank,
    const std::vector<double>& s_values,
    cublasHandle_t cublasHandle,
    cusolverDnHandle_t cusolverHandle,
    curandGenerator_t curandGen,
    unsigned long long seed1,
    unsigned long long seed2)
{
    if (rank == 0) {
        // Zero matrix
        return thrust::device_vector<double>(m * n, 0.0);
    }
    
    if (s_values.size() != rank) {
        fprintf(stderr, "Error: s_values size (%zu) does not match rank (%zu)\n",
                s_values.size(), rank);
        exit(1);
    }
    
    // Allocate Q1 (m x rank) and Q2 (n x rank)
    thrust::device_vector<double> d_Q1(m * rank);
    thrust::device_vector<double> d_Q2(n * rank);
    
    // Generate random orthogonal matrices
    generate_random_orthogonal_matrix(d_Q1.data().get(), m, rank,
                                       cusolverHandle, curandGen, seed1);
    generate_random_orthogonal_matrix(d_Q2.data().get(), n, rank,
                                       cusolverHandle, curandGen, seed2);
    
    // Copy singular values to device
    thrust::device_vector<double> d_S = s_values;
    
    // Compute A = Q1 * diag(S) * Q2^T
    // Step 1: Temp = Q1 * diag(S)  (scale columns of Q1 by S)
    thrust::device_vector<double> d_Temp(m * rank);
    dim3 threads(16, 16);
    dim3 blocks((rank + 15) / 16, (m + 15) / 16);
    scale_columns_kernel<<<blocks, threads>>>(
        d_Temp.data().get(), m,
        d_Q1.data().get(), m,
        d_S.data().get(),
        m, rank);
    CUDA_CHECK(cudaGetLastError());
    
    // Step 2: A = Temp * Q2^T
    thrust::device_vector<double> d_A(m * n);
    const double one = 1.0, zero = 0.0;
    cublasDgemm(cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                m, n, rank,
                &one,
                d_Temp.data().get(), m,
                d_Q2.data().get(), n,
                &zero,
                d_A.data().get(), m);
    
    return d_A;
}
