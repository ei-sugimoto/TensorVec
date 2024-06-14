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

int main()
{
    bool isPrint = true;
    const int thresholdMatrixSize = 16;
    const int m = 32;
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

    loop(i, m)
    {
        loop(j, m)
        {
            M[i * m + j] = i * m + j;
        }
    }
    if (isPrint)
    {
        space();

        print_matrix(m, m, M, m);
    }

    loop(i, m)
    {
        loop(j, m)
        {
            V[i + j * m] = j;
        }
    }

    if (isPrint)
    {
        space();

        print_matrix(m, m, V, m);
    }

    cudaMalloc((void **)&deviceM, sizeof(typeM) * m * m);
    cudaMalloc((void **)&deviceV, sizeof(typeV) * m * m);
    cudaMalloc((void **)&deviceRes, sizeof(typeV) * m * m);

    cudaMemcpy(deviceM, M, sizeof(typeM) * m * m, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceV, V, sizeof(typeV) * m * m, cudaMemcpyHostToDevice);
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