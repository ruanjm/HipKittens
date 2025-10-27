#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <iomanip>
#include <omp.h>

// Error checking macros
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error in " << __FILE__ << " line " << __LINE__ << ": " \
                      << "code " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CPU reference in column-major to match cuBLAS
void cpu_gemm_col_major(float* a, float* b, float* c, int M, int N, int K) {
    // C = A * B in column-major
    // A is M x K, B is K x N, C is M x N
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                // Column-major indexing
                sum += a[i + k * M] * b[k + j * K];
            }
            c[i + j * M] = sum;
        }
    }
}

void check_result(float* h_C, float* h_C_ref, int M, int N) {
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int error_count = 0;
    const float tolerance = 0.5f; // Large tolerance for FP8
    
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        avg_error += error;
        
        if(error > tolerance) {
            if(error_count < 10) {
                int col = i / M;  // Column-major
                int row = i % M;  // Column-major
                std::cout << "Error at index " << i << " (row " << row << ", col " << col 
                         << "): GPU=" << h_C[i] << " CPU=" << h_C_ref[i] 
                         << " (error: " << error << ")" << std::endl;
            }
            else if(error_count == 10) {
                std::cout << "Too many errors, suppressing further output..." << std::endl;
            }
            error_count++;
        }
        max_error = std::max(max_error, error);
    }
    
    avg_error /= (M * N);
    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Avg error: " << avg_error << std::endl;
    std::cout << "Errors above tolerance (" << tolerance << "): " 
              << error_count << " / " << (M*N) 
              << " (" << (100.0f * error_count / (M*N)) << "%)" << std::endl;
}

void benchmark(int m, int n, int k) {
    // Align dimensions to 16 for tensor cores
    m = (m + 15) & ~15;
    n = (n + 15) & ~15;
    k = (k + 15) & ~15;
    
    std::cout << "=== Benchmarking FP8 GEMM ===" << std::endl;
    std::cout << "Matrix dimensions (aligned): M=" << m << ", N=" << n << ", K=" << k << std::endl;

    // Initialize host memory in COLUMN-MAJOR order
    std::vector<float> h_A(m * k);  // M x K in column-major
    std::vector<float> h_B(k * n);  // K x N in column-major
    std::vector<float> h_D(m * n);
    std::vector<float> h_D_ref(m * n);

    // Initialize with random values in column-major order
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    // Fill A (M x K) in column-major
    for (int col = 0; col < k; ++col) {
        for (int row = 0; row < m; ++row) {
            h_A[row + col * m] = dis(gen) * 0.5f;
        }
    }
    
    // Fill B (K x N) in column-major
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < k; ++row) {
            h_B[row + col * k] = dis(gen) * 0.5f;
        }
    }

    // Convert to FP8
    std::vector<__nv_fp8_e4m3> h_A_fp8(m * k);
    std::vector<__nv_fp8_e4m3> h_B_fp8(k * n);
    
    for (int i = 0; i < m * k; ++i) {
        h_A_fp8[i] = __nv_fp8_e4m3(h_A[i]);
    }
    for (int i = 0; i < k * n; ++i) {
        h_B_fp8[i] = __nv_fp8_e4m3(h_B[i]);
    }

    // Allocate device memory
    __nv_fp8_e4m3 *d_A, *d_B, *d_D;
    __nv_bfloat16 *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_D, m * n * sizeof(__nv_fp8_e4m3)));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A_fp8.data(), m * k * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B_fp8.data(), k * n * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, m * n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemset(d_D, 0, m * n * sizeof(__nv_fp8_e4m3)));

    // Create cuBLAS handle
    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    // Create matrix descriptors for COLUMN-MAJOR layout
    cublasLtMatrixLayout_t matA, matB, matC, matD;
    
    // All matrices in column-major with proper leading dimensions
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA, CUDA_R_8F_E4M3, m, k, m));  // M x K, ld=M
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB, CUDA_R_8F_E4M3, k, n, k));  // K x N, ld=K
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC, CUDA_R_16BF, m, n, m));     // M x N, ld=M
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matD, CUDA_R_8F_E4M3, m, n, m));  // M x N, ld=M

    // Create operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    
    // No transpose needed for column-major C = A * B
    const int32_t transa = CUBLAS_OP_N;
    const int32_t transb = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, 
        CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(int32_t)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, 
        CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(int32_t)));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Allocate workspace
    size_t workspaceSize = 32 * 1024 * 1024;  // 32MB
    void* workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    // Query for best algorithm
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, 
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
    
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        handle, operationDesc, matA, matB, matC, matD, 
        preference, 1, &heuristicResult, &returnedResults
    ));
    
    if (returnedResults == 0) {
        std::cerr << "No algorithm found for this configuration!" << std::endl;
        exit(1);
    }
    
    std::cout << "Using algorithm with " << heuristicResult.wavesCount 
              << " waves" << std::endl;

    // Warmup runs
    std::cout << "Running warmup..." << std::endl;
    for (int i = 0; i < 50; ++i) {
        CHECK_CUBLAS(cublasLtMatmul(
            handle,
            operationDesc,
            &alpha,
            d_A, matA,
            d_B, matB,
            &beta,
            d_C, matC,
            d_D, matD,
            &heuristicResult.algo,
            workspace,
            workspaceSize,
            0
        ));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark runs
    const int NUM_ITERATIONS = 100;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << "Running benchmark (" << NUM_ITERATIONS << " iterations)..." << std::endl;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        CHECK_CUBLAS(cublasLtMatmul(
            handle,
            operationDesc,
            &alpha,
            d_A, matA,
            d_B, matB,
            &beta,
            d_C, matC,
            d_D, matD,
            &heuristicResult.algo,
            workspace,
            workspaceSize,
            0
        ));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time = milliseconds / NUM_ITERATIONS;

    // Calculate TFLOPS
    double num_ops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    double seconds = avg_time / 1000.0;
    double tflops = (num_ops / seconds) / 1e12;

    std::cout << "\nResults:" << std::endl;
    std::cout << "Average time: " << std::fixed << std::setprecision(3) 
              << avg_time << " ms" << std::endl;
    std::cout << "Performance: " << std::fixed << std::setprecision(2) 
              << tflops << " TFLOPS" << std::endl;

    // Verify correctness
    std::cout << "\nVerifying correctness..." << std::endl;
    
    // Compute CPU reference in column-major
    cpu_gemm_col_major(h_A.data(), h_B.data(), h_D_ref.data(), m, n, k);

    // Copy GPU result back
    std::vector<__nv_fp8_e4m3> h_D_fp8(m * n);
    CHECK_CUDA(cudaMemcpy(h_D_fp8.data(), d_D, m * n * sizeof(__nv_fp8_e4m3), 
                          cudaMemcpyDeviceToHost));

    // Convert FP8 to float
    for (int i = 0; i < m * n; i++) {
        h_D[i] = float(h_D_fp8[i]);
    }

    // Compare results
    check_result(h_D.data(), h_D_ref.data(), m, n);

    // Cleanup
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matA));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matB));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matC));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matD));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
    CHECK_CUBLAS(cublasLtDestroy(handle));
    
    std::cout << "----------------------------------------\n" << std::endl;
}

int main(int argc, char** argv) {
    // Check GPU
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Running on: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl << std::endl;
    
    // Benchmark different matrix sizes
    std::vector<int> sizes = {2048, 4096, 8192};
    
    for (int size : sizes) {
        benchmark(size, size, size);
    }

    return 0;
}