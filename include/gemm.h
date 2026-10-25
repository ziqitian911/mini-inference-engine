#pragma once
#include "tensor.h"

enum class MatmulMode
{
    Naive,
    Reorder,
    Blocked,
    Threaded
};

Tensor matmul(
    const Tensor &A,
    const Tensor &B,
    MatmulMode mode = MatmulMode::Reorder,
    int BS = 64);