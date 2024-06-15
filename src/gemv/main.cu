#define ll long long
#define loop(i, n) for (ll i = 0; i < n; i++)
#define space() cout << "===============================================" << endl

typedef float typeM;
typedef float typeV;
using namespace std;

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include "../utils/print_matrix.h"
#include "mm_gpu80.cuh"
#include "../utils/timer.hpp"
#include "../utils/operate_matrix.cuh"

int main()
{
    bool isPrint = true;
    const int thresholdMatrixSize = 16;
    const int m = 8;
    class Timer timer;

    typeM *deviceM, *M;
    typeV *deviceV, *deviceRes, *V, *res;

    M = (typeM *)malloc(sizeof(typeM) * m * m);
    V = (typeV *)malloc(sizeof(typeV) * m * m);
    res = (typeV *)malloc(sizeof(typeV) * m * m);

    if (m < thresholdMatrixSize)
    {
        isPrint = true;
    }
    else
    {
        isPrint = false;
    }

    cudaMalloc((void **)&deviceM, sizeof(typeM) * m * m);
    cudaMalloc((void **)&deviceV, sizeof(typeV) * m * m);
    cudaMalloc((void **)&deviceRes, sizeof(typeV) * m * m);

    create_matrix<typeM><<<dim3((m + 16 - 1) / 16, (m + 16 - 1) / 16), dim3(16, 16)>>>(deviceM, m, m);
    create_vector_for_cutlass<typeV><<<dim3((m + 16 - 1) / 16, (m + 16 - 1) / 16), dim3(16, 16)>>>(deviceV, m, m);

    cudaMemcpy(M, deviceM, sizeof(typeM) * m * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(V, deviceV, sizeof(typeV) * m * m, cudaMemcpyDeviceToHost);

    if (isPrint)
    {
        print_matrix(m, m, M, m);
        space();
        print_matrix(m, m, V, m);
    }

    timer.reset();
    mm_gpu<typeM, typeV>(m, m, m, 1.0f, deviceM, deviceV, 1.0f, deviceRes);
    timer.stop();
    space();
    timer.print();

    cudaMemcpy(res, deviceRes, sizeof(typeV) * m * m, cudaMemcpyDeviceToHost);

    if (isPrint)
    {
        space();

        print_matrix(m, m, res, m);
    }
    return 0;
}