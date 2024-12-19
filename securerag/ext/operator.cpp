#include "ATen/ops/matmul.h"
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/from_blob.h>
#include <c10/util/ArrayRef.h>
#include <torch/extension.h>

#include "sgxsecurerag.h"

#include <torch/types.h>
#include <vector>

namespace SecureRAGExtension {
at::Tensor doSecureLinear(const at::Tensor &input, const at::Tensor &weight,
                          const at::Tensor &bias) {
    input.contiguous();
    weight.contiguous();
    bias.contiguous();
    float *inputp = (float *)input.const_data_ptr();
    float *weightp = (float *)weight.const_data_ptr();
    float *biasp = (float *)bias.const_data_ptr();
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
    float *output = (float *)outTensor.mutable_data_ptr();
    sgxSecureLinear(inputp, weightp, biasp, output, N, indim, outdim);
    return outTensor;
}

void doOpenSGX() {
    printf("SGX INITING...\n");
    initSGX();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(SecureRAGExtension, m) {
    m.def("secureLinear(Tensor input, Tensor weight, Tensor bias) -> Tensor");
    m.def("openSGX", &doOpenSGX);
}

TORCH_LIBRARY_IMPL(SecureRAGExtension, CPU, m) {
    m.impl("secureLinear", &doSecureLinear);
}
} // namespace SecureRAGExtension