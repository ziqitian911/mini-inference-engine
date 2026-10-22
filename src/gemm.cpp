#include "gemm.h"

/**
 * @brief Matrix multiply: C = A * B (naive triple-loop GEMM).
 * @param A left matrix, shape M x K.
 * @param B right matrix, shape K x N (A's cols must equal B's rows).
 * @return the product, shape M x N.
 */
Tensor matmul(const Tensor &A, const Tensor &B, bool reorder)
{
    int M = A.dim(0), K = A.dim(1), N = B.dim(1);

    Tensor C({M, N});

    std::fill(C.begin(), C.end(), 0.0f);

    if (reorder)
    {
        // loop reorder
        for (int i = 0; i < M; i++)
        {
            for (int k = 0; k < K; k++)
            {
                for (int j = 0; j < N; j++)
                {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                for (int k = 0; k < K; k++)
                {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    return C;
}
