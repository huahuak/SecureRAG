#include "data.h"

#include <cstddef>
#include <memory>

std::shared_ptr<std::byte> Tensor::dataPtr() { return mem; }

const Tensor Tensor::EMPTY = Tensor(nullptr, 0, Typ::INT, {});