#include "gemm.h"

#include <algorithm>

/**
 * @brief Matrix multiply: C = A * B (naive triple-loop GEMM).
 * @param A left matrix, shape M x K.
 * @param B right matrix, shape K x N (A's cols must equal B's rows).
 * @return the product, shape M x N.
 */
Tensor matmul(const Tensor &A, const Tensor &B, MatmulMode mode, int BS)
{
    int M = A.dim(0), K = A.dim(1), N = B.dim(1);

    Tensor C({M, N});

    std::fill(C.begin(), C.end(), 0.0f);

    const float *__restrict__ a = A.begin();
    const float *__restrict__ b = B.begin();

    float *__restrict__ c = C.begin();

    switch (mode)
    {
    case MatmulMode::Naive:
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                for (int k = 0; k < K; k++)
                {
                    sum += a[i * K + k] * b[k * N + j];
                }
                c[i * N + j] = sum;
            }
        }
        break;

    case MatmulMode::Reorder:
        // loop reorder
        for (int i = 0; i < M; i++)
        {
            for (int k = 0; k < K; k++)
            {
                for (int j = 0; j < N; j++)
                {
                    c[i * N + j] += a[i * K + k] * b[k * N + j];
                }
            }
        }
        break;

    case MatmulMode::Blocked:

        for (int ii = 0; ii < M; ii += BS)
        {
            for (int jj = 0; jj < N; jj += BS)
            {
                for (int kk = 0; kk < K; kk += BS)
                {
                    for (int i = ii; i < std::min(ii + BS, M); i++)
                    {
                        for (int k = kk; k < std::min(kk + BS, K); k++)
                        {
                            for (int j = jj; j < std::min(jj + BS, N); j++)
                            {
                                c[i * N + j] += a[i * K + k] * b[k * N + j];
                            }
                        }
                    }
                }
            }
        }

        break;
    }

    return C;
}
