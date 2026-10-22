#pragma once
#include "tensor.h"

Tensor matmul(const Tensor &A, const Tensor &B, bool reorder = true);