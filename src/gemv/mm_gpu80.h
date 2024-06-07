#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"

#include <iostream>
#include <vector>

template <class ElementAB, class ElementC>
void mm_gpu(int m, int n, int k, ElementC alpha, const ElementAB *d_A, const ElementAB *d_B, ElementC beta, ElementC *d_C)
/*
  m行k列の行列Aとk行n列の行列Bに対して C = αAB + βC をCUTLASSライブラリを用いてGPU上で計算する．
  ElementABは行列A, Bの要素の型，ElementCは行列Cの要素の型であり，
  d_A，d_B，d_Cは，行優先で行列A, B, Cが格納されているデバイスメモリの先頭アドレスである．
　C≠AかつC≠Bでなければならない．

  Tensor Coreを使う場合，nとkには以下の制約がある：
  ----------------------------------------
  ElementAB      n,kの制約
  ----------------------------------------
  double         2の倍数でなければならない
  float          4の倍数でなければならない
  half_t         8の倍数でなければならない

  ----------------------------------------
*/
{
    using RowMajor = cutlass::layout::RowMajor;
    using columnMajor = cutlass::layout::ColumnMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<
        ElementAB, // ElementA
        RowMajor,  // LayoutA
        ElementAB, // ElementB
        RowMajor,  // LayoutB
        ElementC,  // ElementOutput
        RowMajor,  // LayoutOutput
        ElementC   // ElementAccumulator
        >;

    CutlassGemm gemm_operator;

    int lda = k, ldb = n, ldc = n;

    cutlass::Status status = gemm_operator({
        {m, n, k},    // Gemm Problem dimensions
        {d_A, lda},   // Tensor-ref for source matrix A
        {d_B, ldb},   // Tensor-ref for source matrix B
        {d_C, ldc},   // Tensor-ref for source matrix C
        {d_C, ldc},   // Tensor-ref for destination matrix D (may be different memory than source C matrix)
        {alpha, beta} // Scalars used in the Epilogue
    });
    assert(status == cutlass::Status::kSuccess);
}

template <>
void mm_gpu(int m, int n, int k, cutlass::half_t alpha, const cutlass::half_t *d_A, const cutlass::half_t *d_B, cutlass::half_t beta, cutlass::half_t *d_C)
{
    using RowMajor = cutlass::layout::RowMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,                // ElementA
        RowMajor,                       // LayoutA
        cutlass::half_t,                // ElementB
        RowMajor,                       // LayoutB
        cutlass::half_t,                // ElementOutput
        RowMajor,                       // LayoutOutput
        cutlass::half_t,                // ElementAccumulator
        cutlass::arch::OpClassTensorOp, // tag indicating Tensor Cores
        cutlass::arch::Sm80             // tag indicating target GPU compute architecture
        >;

    CutlassGemm gemm_operator;

    int lda = k, ldb = n, ldc = n;
    cutlass::Status status = gemm_operator({
        {m, n, k},    // Gemm Problem dimensions
        {d_A, lda},   // Tensor-ref for source matrix A
        {d_B, ldb},   // Tensor-ref for source matrix B
        {d_C, ldc},   // Tensor-ref for source matrix C
        {d_C, ldc},   // Tensor-ref for destination matrix D (may be different memory than source C matrix)
        {alpha, beta} // Scalars used in the Epilogue
    });
    assert(status == cutlass::Status::kSuccess);
}

template <>
void mm_gpu(int m, int n, int k, float alpha, const cutlass::half_t *d_A, const cutlass::half_t *d_B, float beta, float *d_C)
{
    using RowMajor = cutlass::layout::RowMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,                // ElementA
        RowMajor,                       // LayoutA
        cutlass::half_t,                // ElementB
        RowMajor,                       // LayoutB
        float,                          // ElementOutput
        RowMajor,                       // LayoutOutput
        float,                          // ElementAccumulator
        cutlass::arch::OpClassTensorOp, // tag indicating Tensor Cores
        cutlass::arch::Sm80             // tag indicating target GPU compute architecture
        >;

    CutlassGemm gemm_operator;

    int lda = k, ldb = n, ldc = n;
    cutlass::Status status = gemm_operator({
        {m, n, k},    // Gemm Problem dimensions
        {d_A, lda},   // Tensor-ref for source matrix A
        {d_B, ldb},   // Tensor-ref for source matrix B
        {d_C, ldc},   // Tensor-ref for source matrix C
        {d_C, ldc},   // Tensor-ref for destination matrix D (may be different memory than source C matrix)
        {alpha, beta} // Scalars used in the Epilogue
    });
    assert(status == cutlass::Status::kSuccess);
}