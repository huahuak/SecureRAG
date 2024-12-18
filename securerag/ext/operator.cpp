#include "ATen/ops/matmul.h"
#include <ATen/ops/from_blob.h>
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
    int N = input.size(0);
    int indim = input.size(1);
    int outdim = weight.size(0);
    float *output = (float *)malloc(N * outdim * sizeof(float));
    sgxSecureLinear(inputp, weightp, biasp, output, N, indim, outdim);
    at::Tensor outTensor =
        torch::from_blob((void *)output, {N, outdim}, torch::kFloat32);
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