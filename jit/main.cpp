#include <ATen/core/interned_strings.h>
#include <torch/extension.h>
#include <torch/script.h>  // TorchScript header
#include <torch/script.h>

#include <iostream>

torch::Tensor my_custom_bmm(const at::Tensor & self, const at::Tensor & mat2) {
    std::cout << "[INFO] Running custom BMM" << std::endl;
    return at::bmm(self, mat2);
}

TORCH_LIBRARY_IMPL(aten, CPU, m) { m.impl("bmm", my_custom_bmm); }

int do_main() {
    torch::jit::script::Module module;

    try {
        module = torch::jit::load("jit/attn.pt");
        int embed_dim = 768;
        int bsz = 8;
        int tgtlen = 4;

        // 构造输入
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::randn({tgtlen, bsz, embed_dim}));
        inputs.push_back(torch::randn({tgtlen, bsz, embed_dim}));

        // 执行模型
        at::Tensor output = module.forward(inputs).toTensor();

        // std::cout << output << std::endl;

        auto graph = module.get_method("forward").graph();
        // 遍历所有节点
        for (auto node : graph->nodes()) {
            std::cout << node->kind().toDisplayString() << std::endl;
        }

    } catch (const c10::Error& e) {
        std::cerr << "加载模型失败: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

int main() {
    do_main();
}