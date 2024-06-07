#include <iostream>
#include <vector>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "gemv/gemv.cuh"

#define ll long long
#define loop(i, n) for (ll i = 0; i < n; i++)

typedef float typeM;
typedef float typeV;
using namespace std;

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils/cublas_utils.h"

using data_type = double;

int main(int argc, char *argv[])
{
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = 8;
    const int n = 8;
    const int lda = m;

    /*
     *   A = | 1.0 2.0 |
     *       | 3.0 4.0 |
     *   x = | 5.0 6.0 |
     */

    std::vector<data_type> A(m * n, 0); // サイズを m*n に設定し、全要素を0で初期化
    std::vector<data_type> x(n, 0);     // サイズを n に設定し、全要素を0で初期化
    std::vector<data_type> y(m, 0);
    const data_type alpha = 1.0;
    const data_type beta = 0.0;
    const int incx = 1;
    const int incy = 1;

    loop(i, m)
    {
        loop(j, n)
        {
            A[i * m + j] = i * n + j;
        }
    }

    loop(i, n)
    {
        x.at(i) = i + 1;
    }

    data_type *d_A = nullptr;
    data_type *d_x = nullptr;
    data_type *d_y = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;

    printf("A\n");
    print_matrix(m, n, A.data(), lda);
    printf("=====\n");

    printf("x\n");
    print_vector(x.size(), x.data());
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(data_type) * x.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(data_type) * y.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_x, x.data(), sizeof(data_type) * x.size(), cudaMemcpyHostToDevice,
                               stream));

    /* step 3: compute */
    CUBLAS_CHECK(
        cublasDgemv(cublasH, transa, m, n, &alpha, d_A, lda, d_x, incx, &beta, d_y, incy));

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(y.data(), d_y, sizeof(data_type) * y.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   y = | 17.00 39.00 |
     */

    printf("y\n");
    print_vector(y.size(), y.data());
    printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}