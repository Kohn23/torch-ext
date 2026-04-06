#include <iostream>

#include "qbmatmul.h"
#include "common/cucheck.h"
#include "common/matgen.h"
#include "common/numerr.h"

void test_qb_factorization() {
    auto d_A = generate_low_rank_matrix(m, n, rank, s_values, cublasHandle, cusolverHandle, curandGen);
    
    