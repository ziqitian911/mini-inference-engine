#include "gemm.h"

/**
 * @brief Matrix multiply: C = A * B (naive triple-loop GEMM).
 * @param A left matrix, shape M x K.
 * @param B right matrix, shape K x N (A's cols must equal B's rows).
 * @return the product, shape M x N.
 */
Tensor matmul(const Tensor &A, const Tensor &B)
{
    int M = A.dim(0), K = A.dim(1), N = B.dim(1);

    Tensor C({M, N});

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
            {
                sum += A.at({i, k}) * B.at({k, j});
            }
            C.at({i, j}) = sum;
        }
    }

    return C;
}
