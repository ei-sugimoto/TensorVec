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

using data_type = double;

int main(int argc, char *argv[])
{
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = 32;
    const int n = 32;
    const int lda = m;
    Timer timer;

    std::vector<data_type> A(m * n, 0);
    std::vector<data_type> x(n, 0);
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
        x.at(i) = i;
    }

    data_type *d_A = nullptr;
    data_type *d_x = nullptr;
    data_type *d_y = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;

    printf("A\n");
    print_matrix(m, n, A.data(), lda);
    space();

    printf("x\n");
    print_vector(x.size(), x.data());
    space();

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

    timer.reset();
    CUBLAS_CHECK(
        cublasDgemv(cublasH, transa, m, n, &alpha, d_A, lda, d_x, incx, &beta, d_y, incy));
    timer.stop();
    timer.print();
    space();

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(y.data(), d_y, sizeof(data_type) * y.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   y = | 17.00 39.00 |
     */

    printf("y\n");
    print_vector(y.size(), y.data());
    space();
    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}