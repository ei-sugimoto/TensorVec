# TensorVec
TensorVec is GEMV with cutlass. 

generaly, cutlass is not support vector. Because the Tensor Cores used in cutlass are support at least 2x2 matrix.
But Tensor Cores are most powerful to multiply matrix in CUDA Core. 
This Repository validate to GEMV. 

*GEMV is General multiply Matrix Vector.