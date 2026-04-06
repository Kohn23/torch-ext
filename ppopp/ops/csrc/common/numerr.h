#include <iostream>
#include <type_traits>

#include <cublas_v2.h>
#include <thrust/device_vector.h>

template <typename T>
class Error {
public:
    T abs_error;
    T rel_error;

    Error(T abs_err = T(0), T rel_err = T(0))
        : abs_error(abs_err), rel_error(rel_err) {}

};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Error<T>& err) {
    os << "Absolute error: " << err.abs_error
       << ", Relative error: " << err.rel_error;
    return os;
}



/** Compute the norm2 (Frobenius norm) of the difference between two matrices 
 * 
 * @param handle cuBLAS handle
 * @param d_A pointer to the first matrix on device
 * @param d_B pointer to the second matrix on device
 * @param m number of rows
 * @param n number of columns
 * @return the Frobenius norm of the difference
 */
template <typename T>
Error<T> norm2_error(cublasHandle_t handle,
                              const T* d_A,
                              const T* d_B,
                              int m, int n) {
    size_t total = (size_t)m * (size_t)n;
    if (total == 0) {
        return Error<T>(T(0), T(0));
    }

    thrust::device_vector<T> d_diff(total);
    T* d_diff_ptr = thrust::raw_pointer_cast(d_diff.data());

    T norm_A = 0;
    T resid = 0;

    if constexpr (std::is_same_v<T, float>) {
        cublasSnrm2(handle, (int)total, d_A, 1, &norm_A);
        T alpha = 1.0f, beta = -1.0f;
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, &alpha, d_A, m, &beta, d_B, m, d_diff_ptr, m);
        cublasSnrm2(handle, (int)total, d_diff_ptr, 1, &resid);
    } else if constexpr (std::is_same_v<T, double>) {
        cublasDnrm2(handle, (int)total, d_A, 1, &norm_A);
        T alpha = 1.0, beta = -1.0;
        cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, &alpha, d_A, m, &beta, d_B, m, d_diff_ptr, m);
        cublasDnrm2(handle, (int)total, d_diff_ptr, 1, &resid);
    } else {
        static_assert(sizeof(T) == 0, "Unsupported type for frobenius_norm_error");
    }

    const T eps = (std::is_same_v<T, float>) ? T(1e-30) : T(1e-30);
    T rel = (norm_A > eps) ? (resid / norm_A) : resid;

    return Error<T>(resid, rel);
}

