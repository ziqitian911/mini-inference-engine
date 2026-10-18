#pragma once
#include "tensor.h"

Tensor relu(const Tensor &input);
Tensor linear(const Tensor &input, const Tensor &weights, const Tensor &bias);
Tensor softmax(const Tensor &input);