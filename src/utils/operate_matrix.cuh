template <typename T>
__global__ void create_matrix(T *matrix, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < rows && idy < cols)
    {
        matrix[idy * cols + idx] = (T)(idy * cols + idx);
    }
}

template <typename T>
__global__ void create_vector(T *vector, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        vector[idx] = (T)(idx);
    }
}

template <typename T>
__global__ void create_vector_for_cutlass(T *vector, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < rows && idy < cols)
    {
        vector[idy * cols + idx] = (T)(idy);
    }
}

template <typename T>
__global__ void clear_matrix(T *matrix, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < rows && idy < cols)
    {
        matrix[idy * cols + idx] = 0;
    }
}
