


#include <stdio.h>
#include <stdlib.h> 

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSOLVER error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CURAND_CHECK(call) \
    do { \
        curandStatus_t status = call; \
        if (status != CURAND_STATUS_SUCCESS) { \
            fprintf(stderr, "cuRAND error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
