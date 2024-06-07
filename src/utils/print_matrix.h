#include <iostream>
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