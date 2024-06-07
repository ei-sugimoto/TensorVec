#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_array.h"

#include <iostream>
#include <vector>

template <typename typeM, typename typeV>
void gemv(typeM *M, typeV *V, int m, int k, typeM *res)
{
    using rowMajor = cutlass::layout::RowMajor;
    using columnMajor = cutlass::layout::ColumnMajor;

    using cutlassGemv = cutlass::gemm::device::Gemm<
        typeM,
        rowMajor,
        typeV,
        columnMajor,
        typeM,
        rowMajor>;

    cutlassGemv gemv;

    cutlass::Status status = gemv({{m, k, 1},
                                   {M, m},
                                   {V, k},
                                   {res, m},
                                   {res, m},
                                   {1, 0}});
    assert(status == cutlass::Status::kSuccess);
}