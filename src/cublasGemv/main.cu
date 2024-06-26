#define ll long long
#define loop(i, n) for (ll i = 0; i < n; i++)
#define space() cout << "===============================================" << endl

typedef float typeM;
typedef float typeV;
using namespace std;

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../utils/cublas_utils.h"
#include "../utils/timer.hpp"
#include "../utils/operate_matrix.cuh"

using data_type = double;

int main(int argc, char *argv[])
{
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    bool isPrint = true;
    const int thresholdMatrixSize = 16;

    const int m = 4352;
    const int n = 4352;
    const int lda = m;
    Timer timer;

    std::vector<data_type> A(m * n, 0);
    std::vector<data_type> x(n, 0);
    std::vector<data_type> y(m, 0);
    const data_type alpha = 1.0;
    const data_type beta = 0.0;
    const int incx = 1;
    const int incy = 1;

    if (m < thresholdMatrixSize)
    {
        isPrint = true;
    }
    else
    {
        isPrint = false;
    }

    data_type *d_A = nullptr;
    data_type *d_x = nullptr;
    data_type *d_y = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(data_type) * x.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(data_type) * y.size()));

    create_matrix<data_type><<<dim3((m + 16 - 1) / 16, (n + 16 - 1) / 16), dim3(16, 16)>>>(d_A, m, n);
    create_vector<data_type><<<dim3((n + 16 - 1) / 16, (1 + 16 - 1) / 16), dim3(16, 16)>>>(d_x, n);
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(x.data(), d_x, sizeof(data_type) * x.size(), cudaMemcpyDeviceToHost));

    if (isPrint)
    {
        space();
        printf("A\n");
        print_matrix(m, n, A.data(), lda);
        space();
    }

    if (isPrint)
    {
        space();
        printf("x\n");
        print_vector(x.size(), x.data());
        space();
    }

    CUBLAS_CHECK(cublasCreate(&cublasH));
    cublasOperation_t transa = CUBLAS_OP_T;

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    timer.reset();
    CUBLAS_CHECK(
        cublasDgemv(cublasH, transa, m, n, &alpha, d_A, lda, d_x, incx, &beta, d_y, incy));
    cudaDeviceSynchronize();

    timer.stop();
    space();
    cout << "m : " << m << endl;
    timer.print();
    space();

    CUDA_CHECK(cudaMemcpyAsync(y.data(), d_y, sizeof(data_type) * y.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (isPrint)
    {
        space();
        printf("y\n");
        print_vector(y.size(), y.data());
        space();
    }
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
