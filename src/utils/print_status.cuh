#include <cutlass/cutlass.h>
#include <string>

std::string CutlassStatusToString(cutlass::Status status)
{
    switch (status)
    {
    case cutlass::Status::kSuccess:
        return "Success";
    case cutlass::Status::kErrorMisalignedOperand:
        return "Error: Misaligned Operand";
    case cutlass::Status::kErrorInvalidLayout:
        return "Error: Invalid Layout";
    case cutlass::Status::kErrorInvalidDataType:
        return "Error: Invalid Data Type";
    case cutlass::Status::kErrorInvalidProblem:
        return "Error: Invalid Problem";
    case cutlass::Status::kErrorNotSupported:
        return "Error: Not Supported";
    case cutlass::Status::kErrorWorkspaceNull:
        return "Error: Workspace Null";
    case cutlass::Status::kErrorInternal:
        return "Error: Internal";
    case cutlass::Status::kErrorMemoryAllocation:
        return "Error: Memory Allocation";
    case cutlass::Status::kInvalid:
        return "Error: Invalid";
    default:
        return "Error: Unknown Status";
    }
}