#include <iostream>
#include "cutlass/half.h"
template <typename T>
void print_matrix(const int &m, const int &n, const T *A, const int &lda);

template <>
void print_matrix(const int &m, const int &n, const float *A, const int &lda)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::printf("%0.2f ", A[i * lda + j]);
        }
        std::printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const double *A, const int &lda)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::printf("%0.2f ", A[i * lda + j]);
        }
        std::printf("\n");
    }
}
void print_matrix(int const &height, int const &width, cutlass::half_t const *matrix, int const &lda)
{
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            std::cout << static_cast<float>(matrix[i * lda + j]) << " ";
        }
        std::cout << std::endl;
    }
}