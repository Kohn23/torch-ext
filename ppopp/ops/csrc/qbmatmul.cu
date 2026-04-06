#include "qbmatmul.h"

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
) {
    const double one = 1.0;
    const double zero = 0.0;
    const double minus_one = -1.0;
    
    int accumulated_rank = 0;
    bool stop_condition_met = false;
    int final_rank = 0;
    
    // 创建 A 的备份（用于列筛选时读取原始数据）
    thrust::device_vector<double> d_A_backup(m * n);
    CUDA_CHECK(cudaMemcpy(d_A_backup.data().get(), d_A_ptr, 
                          m * n * sizeof(double), cudaMemcpyDeviceToDevice));
    double* d_A_backup_ptr = thrust::raw_pointer_cast(d_A_backup.data());
    
    // 用于存储过滤后的 A 矩阵
    double* d_A_filtered_ptr = d_A_ptr;  // 原地更新
    
    int block_size = 64;  // 分块大小，可根据性能调整
    
    for (int j = 0; j < (n + block_size - 1) / block_size; ++j) {
        int col_start = j * block_size;
        int row_start = j * block_size;
        int panel_height = m - row_start;
        int panel_width = std::min(block_size, n - col_start);
        panel_width = std::min(panel_width, k - accumulated_rank);
        
        if (panel_height <= 0 || panel_width <= 0) break;
        
        // ====================================================================
        // 步骤 1: 对当前面板进行 QR 分解
        // ====================================================================
        thrust::device_vector<double> d_R_tsqr(panel_width * panel_width);
        double* d_R_tsqr_ptr = thrust::raw_pointer_cast(d_R_tsqr.data());
        
        double* d_panel_ptr = d_A_ptr + col_start * m + row_start;
        
        thrust::device_vector<double> d_tau(panel_width);
        double* d_tau_ptr = thrust::raw_pointer_cast(d_tau.data());
        
        thrust::device_vector<int> d_devInfo(1);
        int* d_devInfo_ptr = thrust::raw_pointer_cast(d_devInfo.data());
        
        int lwork_geqrf = 0;
        int lwork_orgqr = 0;
        
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cusolverHandle, panel_height, panel_width,
                                                    d_panel_ptr, m, &lwork_geqrf));
        CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(cusolverHandle, panel_height, panel_width, panel_width,
                                                    d_panel_ptr, m, d_tau_ptr, &lwork_orgqr));
        
        int lwork_qr = std::max(lwork_geqrf, lwork_orgqr);
        thrust::device_vector<double> d_qr_work(lwork_qr);
        double* d_qr_work_ptr = thrust::raw_pointer_cast(d_qr_work.data());
        
        // 执行 QR 分解
        CUSOLVER_CHECK(cusolverDnDgeqrf(cusolverHandle, panel_height, panel_width,
                                         d_panel_ptr, m,
                                         d_tau_ptr,
                                         d_qr_work_ptr, lwork_qr,
                                         d_devInfo_ptr));
        
        // 提取 R 的上三角部分
        dim3 threads_R_extract(16, 16);
        dim3 blocks_R_extract((panel_width + 15) / 16, (panel_width + 15) / 16);
        extract_upper_triangle_kernel<<<blocks_R_extract, threads_R_extract>>>(
            d_R_tsqr_ptr, panel_width,
            d_panel_ptr, m,
            panel_width, panel_width);
        CUDA_CHECK(cudaGetLastError());
        
        // 显式形成 Q
        CUSOLVER_CHECK(cusolverDnDorgqr(cusolverHandle, panel_height, panel_width, panel_width,
                                         d_panel_ptr, m,
                                         d_tau_ptr,
                                         d_qr_work_ptr, lwork_qr,
                                         d_devInfo_ptr));
        
        // ====================================================================
        // 步骤 2: 提取 R 的对角元并筛选重要列
        // ====================================================================
        thrust::device_vector<double> diag_elements_gpu(panel_width);
        extractDiagonal<<<(panel_width + 255) / 256, 256>>>(
            thrust::raw_pointer_cast(diag_elements_gpu.data()),
            d_R_tsqr_ptr,
            panel_width,
            0, panel_width);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        thrust::host_vector<double> diag_elements_cpu(panel_width);
        thrust::copy(diag_elements_gpu.begin(), diag_elements_gpu.end(), 
                     diag_elements_cpu.begin());
        
        std::vector<int> h_keep_indices;
        h_keep_indices.reserve(panel_width);
        
        double max_diag = 0.0;
        for (int i = 0; i < panel_width; ++i) {
            double abs_val = std::abs(diag_elements_cpu[i]);
            if (abs_val > max_diag) {
                max_diag = abs_val;
            }
        }
        
        double relative_eps = max_diag * eps;
        for (int i = 0; i < panel_width; ++i) {
            if (std::abs(diag_elements_cpu[i]) >= relative_eps) {
                h_keep_indices.push_back(i);
            }
        }
        
        int new_panel_width = h_keep_indices.size();
        if (new_panel_width == 0) {
            stop_condition_met = true;
            final_rank = accumulated_rank;
            break;
        }
        
        // 记录全局列索引
        for (int idx : h_keep_indices) {
            h_keep_cols_global.push_back(col_start + idx);
        }
        
        // ====================================================================
        // 步骤 3: 筛选保留的列
        // ====================================================================
        thrust::device_vector<int> d_keep_indices = h_keep_indices;
        int* d_keep_indices_ptr = thrust::raw_pointer_cast(d_keep_indices.data());
        
        // 提取 Q_filtered (保留的 Q 列)
        thrust::device_vector<double> d_Q_filtered(panel_height * new_panel_width);
        double* d_Q_filtered_ptr = thrust::raw_pointer_cast(d_Q_filtered.data());
        
        dim3 threads_gather(16, 16);
        dim3 blocks_gather((new_panel_width + 15) / 16, (panel_height + 15) / 16);
        select_columns_kernel<<<blocks_gather, threads_gather>>>(
            d_Q_filtered_ptr, panel_height,
            d_A_ptr + col_start * m + row_start, m,
            d_keep_indices_ptr,
            panel_height, new_panel_width);
        CUDA_CHECK(cudaGetLastError());
        
        // 提取 R_filtered (保留的 R 列，然后保留的行)
        thrust::device_vector<double> d_R_temp(panel_width * new_panel_width);
        double* d_R_temp_ptr = thrust::raw_pointer_cast(d_R_temp.data());
        
        dim3 threads_gather_R(16, 16);
        dim3 blocks_gather_R((new_panel_width + 15) / 16, (panel_width + 15) / 16);
        select_columns_kernel<<<blocks_gather_R, threads_gather_R>>>(
            d_R_temp_ptr, panel_width,
            d_R_tsqr_ptr, panel_width,
            d_keep_indices_ptr,
            panel_width, new_panel_width);
        CUDA_CHECK(cudaGetLastError());
        
        thrust::device_vector<double> d_R_filtered(new_panel_width * new_panel_width);
        double* d_R_filtered_ptr = thrust::raw_pointer_cast(d_R_filtered.data());
        
        dim3 threads_gather_R_rows(16, 16);
        dim3 blocks_gather_R_rows((new_panel_width + 15) / 16, (new_panel_width + 15) / 16);
        select_rows_kernel<<<blocks_gather_R_rows, threads_gather_R_rows>>>(
            d_R_filtered_ptr, new_panel_width,
            d_R_temp_ptr, panel_width,
            d_keep_indices_ptr,
            new_panel_width, new_panel_width);
        CUDA_CHECK(cudaGetLastError());
        
        // ====================================================================
        // 步骤 4: 使用显式 Q 进行投影更新 (I - QQ^T)A
        // ====================================================================
        int trailing_matrix_cols = n - (col_start + panel_width);
        if (trailing_matrix_cols > 0) {
            double* d_A_trailing_ptr = d_A_ptr + (col_start + panel_width) * m + row_start;
            
            thrust::device_vector<double> d_Temp1(new_panel_width * trailing_matrix_cols);
            double* d_Temp1_ptr = thrust::raw_pointer_cast(d_Temp1.data());
            
            // T = Q_filtered^T * A_trailing
            CUBLAS_CHECK(cublasDgemm(cublasHandle,
                                      CUBLAS_OP_T, CUBLAS_OP_N,
                                      new_panel_width, trailing_matrix_cols, panel_height,
                                      &one,
                                      d_Q_filtered_ptr, panel_height,
                                      d_A_trailing_ptr, m,
                                      &zero,
                                      d_Temp1_ptr, new_panel_width));
            
            // A_trailing = A_trailing - Q_filtered * T
            CUBLAS_CHECK(cublasDgemm(cublasHandle,
                                      CUBLAS_OP_N, CUBLAS_OP_N,
                                      panel_height, trailing_matrix_cols, new_panel_width,
                                      &minus_one,
                                      d_Q_filtered_ptr, panel_height,
                                      d_Temp1_ptr, new_panel_width,
                                      &one,
                                      d_A_trailing_ptr, m));
        }
        
        // ====================================================================
        // 步骤 5: 存储 Q 和 R 的结果（如果需要）
        // ====================================================================
        if (d_Q_final_ptr != nullptr) {
            CUDA_CHECK(cudaMemcpy2D(d_Q_final_ptr + accumulated_rank * m,
                                     m * sizeof(double),
                                     d_Q_filtered_ptr,
                                     panel_height * sizeof(double),
                                     panel_height * sizeof(double),
                                     new_panel_width,
                                     cudaMemcpyDeviceToDevice));
        }
        
        if (d_B_final_ptr != nullptr) {
            CUDA_CHECK(cudaMemcpy2D(d_B_final_ptr + accumulated_rank * n,
                                     n * sizeof(double),
                                     d_R_filtered_ptr,
                                     new_panel_width * sizeof(double),
                                     new_panel_width * sizeof(double),
                                     new_panel_width,
                                     cudaMemcpyDeviceToDevice));
        }
        
        // ====================================================================
        // 步骤 6: 更新秩并检查停止条件
        // ====================================================================
        accumulated_rank += new_panel_width;
        
        if (new_panel_width < panel_width) {
            stop_condition_met = true;
            final_rank = accumulated_rank;
        }
        
        if (accumulated_rank >= k) {
            stop_condition_met = true;
            final_rank = accumulated_rank;
            printf("\n         !!! Stopping: Reached or exceeded theoretical rank. Final rank set to: %d\n", final_rank);
        }
        
        if (stop_condition_met) {
            break;
        }
    }
    
    if (!stop_condition_met) {
        final_rank = accumulated_rank;
    }
    
    printf("\n========= Rank Revealing QR Complete =========\n");
    printf("Target rank: %d, Actual rank: %d\n", k, final_rank);
    printf("Selected columns: %zu\n", h_keep_cols_global.size());
    printf("===============================================\n");
    
    return final_rank;
}

