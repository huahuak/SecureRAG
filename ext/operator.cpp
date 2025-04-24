#include <ATen/core/TensorBody.h>
#include <ATen/core/jit_type.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/linear.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/intrusive_ptr.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ATen/ops/matmul.h"
#include "data.h"
#include "sgx_securerag.h"

#define FP(tensor) (float *)HelperFunc::getTensorConstPtr<float>(tensor)
#define MFP(tensor) (float *)HelperFunc::getTensorMutablePtr<float>(tensor)

// MARK: HelperFunc
namespace HelperFunc {
template <typename T>
const T *getTensorConstPtr(const at::Tensor &tensor) {
    auto t = tensor;
    if (!t.is_contiguous()) {
        t = t.contiguous();
    }
    return (T *)t.const_data_ptr();
}
template <typename T>
const T *getTensorMutablePtr(const at::Tensor &tensor) {
    auto t = tensor;
    if (!t.is_contiguous()) {
        t = t.contiguous();
    }
    return (T *)t.mutable_data_ptr();
}
}  // namespace HelperFunc

class PytorchTensorRef : public torch::CustomClassHolder {
   public:
    const int64_t id;
    int64_t siz;
    int typ;

    PytorchTensorRef(int64_t id, int64_t siz, int typ)
        : id(id), siz(siz), typ(typ){};

    int64_t getId() { return id; };

    c10::intrusive_ptr<PytorchTensorRef> clone() const {
        return c10::make_intrusive<PytorchTensorRef>(id, siz, typ);
    }

    TensorRef toTensorRef() { return TensorRef(id, siz); }
};

namespace SecureRAGExtension {

c10::intrusive_ptr<PytorchTensorRef> doCopyTensorToSGX(
    const at::Tensor &tensor) {
    TORCH_CHECK(tensor.device().type() == at::DeviceType::CPU,
                "itensor_view_from_dense expects CPU tensor input");
    TORCH_CHECK(tensor.layout() == at::Layout::Strided,
                "itensor_view_from_dense expects dense tensor input");
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float,
                "itensor_view_from_dense expects float tensor input");
    if (tensor.dtype() != torch::kFloat32) {
        printf("[ERROR]: TENSOR IS NOT FLOAT32.");
        exit(-1);
    }
    std::shared_ptr<std::byte> mem =
        std::shared_ptr<std::byte>((std::byte *)FP(tensor), [](std::byte *ptr) {
            // dont free memory which is from pytorch.
        });
    int siz = tensor.element_size() * tensor.numel();
    auto sizes = tensor.sizes();
    Tensor t(mem, siz, Typ::FLOAT32,
             std::vector<long>(sizes.begin(), sizes.end()));
    TensorRef ref = sgxCopyTensorToSGX(t);
    return PytorchTensorRef(ref.id, siz, int(Typ::FLOAT32)).clone();
}

at::Tensor doCopyTensorFromSGX(
    const c10::intrusive_ptr<PytorchTensorRef> &ref) {
    Tensor t = sgxCopyTensorFromSGX(ref->toTensorRef());
    auto p = t.dataPtr();
    at::Tensor ret = at::from_blob(
        t.dataPtr().get(), c10::IntArrayRef(t.dim.data(), t.dim.size()),
        [p](void *ptr) {
            // hold shared_ptr to prolong memory lifecycle.
        },
        torch::kFloat32);
    return ret;
}

at::Tensor doSecureLinear(const at::Tensor &input, const at::Tensor &weight,
                          const at::Tensor &bias) {
    int N = 1;
    if (input.dim() < 3) {
        N = input.size(0);
    } else {
        for (int i = 0; i < input.dim() - 1; i++) {
            N *= input.size(i);
        }
    }
    int indim = input.size(1);
    int outdim = weight.size(0);

    c10::IntArrayRef inshape = input.sizes();
    std::vector<int64_t> shape(inshape.begin(), inshape.end());
    shape[shape.size() - 1] = outdim;
    at::IntArrayRef outshape(shape);
    at::Tensor outTensor = at::empty(outshape, input.options());

    outTensor.contiguous();
    sgxSecureLinear(FP(input), FP(weight), FP(bias), MFP(outTensor), N, indim,
                    outdim);
    return outTensor;
}

at::Tensor doSecureAttention(const at::Tensor &query, const at::Tensor &key,
                             const at::Tensor &qw, const at::Tensor &qb,
                             const at::Tensor &kw, const at::Tensor &kb,
                             const at::Tensor &vw, const at::Tensor &vb,
                             const at::Tensor &outw, const at::Tensor &outb,
                             int dim, int nh) {
    at::Tensor outTensor = at::empty({0});
    int bsz;
    int tgtlen;
    int srclen;
    sgxSecureAttention(FP(query), FP(key), MFP(outTensor), FP(qw), FP(qb),
                       FP(kw), FP(kb), FP(vw), FP(vb), FP(outw), FP(outb), bsz,
                       tgtlen, srclen, dim, nh);
    return outTensor;
}

void doOpenSGX() {
    printf("SGX INITING...\n");
    initSGX();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(SecureRAGExtension, m) {
    m.class_<PytorchTensorRef>("PytorchTensorRef")
        .def("id", &PytorchTensorRef::getId);
    m.def("copyTensorToSGX", doCopyTensorToSGX);
    m.def("copyTensorFromSGX", doCopyTensorFromSGX);
    // m.def("secureLinear(Tensor input, Tensor weight, Tensor bias) ->
    // Tensor");
    m.def("secureLinear", doSecureLinear);
    // m.def(
    //     "secureAttention(Tensor query, Tensor key, Tensor qw, Tensor qb, "
    //     "Tensor kw, Tensor kb, Tensor vw, Tensor vb, Tensor outw, "
    //     "Tensor outb, int dim, int nh) -> Tensor");
    m.def("openSGX", doOpenSGX);
}

// TORCH_LIBRARY_IMPL(SecureRAGExtension, CPU, m) {
//     m.impl("secureLinear", &doSecureLinear);
//     m.impl("secureAttention", &doSecureAttention);
// }
}  // namespace SecureRAGExtension
