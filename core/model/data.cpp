#include "data.h"

#include <cstddef>
#include <memory>

std::shared_ptr<std::byte> Tensor::dataPtr() { return mem; }

size_t Tensor::elementNum() {
    size_t num = 1;
    for (auto d : dim) {
        num *= d;
    }
    return num;
}

Tensor Tensor::convertInt32toTensor(int i) {
    auto mem = (std::byte*)malloc(sizeof(int));
    *(int*)mem = i;
    return Tensor(mem, sizeof(int), Typ::INT32, {1});
}

const Tensor Tensor::EMPTY = Tensor(nullptr, 0, Typ::INT32, {});