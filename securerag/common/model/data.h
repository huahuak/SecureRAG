#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <typeinfo>
#include <unordered_map>
#include <vector>

using ID = size_t;

// MARK: Tensor
enum class Typ {
    INT32,
    FLOAT32,
};

class TensorRef {
   public:
    const ID id;
    const size_t elementSize;

   public:
    TensorRef(ID id, size_t siz) : id(id), elementSize(siz){};
};

class Tensor {
    std::shared_ptr<std::byte> mem;

   public:
    Tensor(void* mem, size_t siz, Typ typ, std::vector<long> dim)
        : mem(std::shared_ptr<std::byte>((std::byte*)mem,
                                         [](std::byte* ptr) { delete[] ptr; })),
          elementSize(siz),
          typ(typ),
          dim(dim){};

    Tensor(std::shared_ptr<std::byte> mem, size_t siz, Typ typ,
           std::vector<long> dim)
        : mem(mem), elementSize(siz), typ(typ), dim(dim){};

    std::shared_ptr<std::byte> dataPtr();
    size_t elementNum();

    template <typename P>
    P first() {
        return (P*)mem.get();
    }

    static Tensor convertInt32toTensor(int i);

    const size_t elementSize;
    const std::vector<long> dim;
    const Typ typ;

    static const Tensor EMPTY;
};

using TensorDict = std::unordered_map<std::string, std::shared_ptr<Tensor>>;

// MARK: DAG

#endif