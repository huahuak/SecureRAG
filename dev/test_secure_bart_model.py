import unittest

import sys, os

sys.path.insert(0, os.getcwd() + "/" + "../src")

import random
import torch
import torch.nn.functional as F
from transformers.modeling_bart import Attention


random.seed(2025)


@unittest.skip("")
class CustomTests(unittest.TestCase):
    @unittest.skip("")
    def test_torchscript(self):
        import torch
        import torch.nn as nn

        # 定义模型
        class MyDynamicModel(nn.Module):
            def __init__(self):
                super(MyDynamicModel, self).__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                if x.sum() > 10:
                    return self.fc(x) * 2
                else:
                    return self.fc(x) / 2

        # 转换为 TorchScript
        model = MyDynamicModel()
        scripted_model = torch.jit.script(model)
        print(scripted_model)
        print(scripted_model.code)
        print(scripted_model.graph)

    @unittest.skip("")
    def test_mm(self):
        import time
        import torch
        import matplotlib.pyplot as plt

        # 设置矩阵尺寸
        M, N, K = 1024, 1024, 1024
        batch_sizes = [64 * i for i in range(1, 30)]  # 从 64 到 64*30 的 batch size

        # 初始化 GPU 最大尺寸矩阵
        max_batch_size = max(batch_sizes)
        A_gpu = torch.randn(max_batch_size, M, K, device="cuda", dtype=torch.float16)
        B_gpu = torch.randn(max_batch_size, K, N, device="cuda", dtype=torch.float16)

        # 初始化 CPU 最大尺寸矩阵
        A_cpu = torch.randn(max_batch_size, M, K, device="cpu", dtype=torch.float16)
        B_cpu = torch.randn(max_batch_size, K, N, device="cpu", dtype=torch.float16)

        # 初始化结果存储
        gpu_time_taken = []
        cpu_time_taken = []

        # 遍历不同的 batch size
        for batch_size in batch_sizes:
            # GPU 计算
            A_sub_gpu = A_gpu[:batch_size]
            B_sub_gpu = B_gpu[:batch_size]

            start_gpu = time.time()
            C_gpu = torch.matmul(A_sub_gpu, B_sub_gpu)
            torch.cuda.synchronize()  # 等待 GPU 计算完成
            end_gpu = time.time()

            gpu_elapsed_time = end_gpu - start_gpu
            gpu_time_taken.append(gpu_elapsed_time)

            # CPU 计算
            A_sub_cpu = A_cpu[:batch_size]
            B_sub_cpu = B_cpu[:batch_size]

            start_cpu = time.time()
            C_cpu = A_sub_cpu + B_sub_cpu
            end_cpu = time.time()

            cpu_elapsed_time = end_cpu - start_cpu
            cpu_time_taken.append(cpu_elapsed_time)

            print(f"Batch size: {batch_size}, GPU Time: {gpu_elapsed_time:.4f} s, CPU Time: {cpu_elapsed_time:.4f} s")

        # 绘制折线图
        plt.figure(figsize=(10, 6), dpi=200)
        plt.plot(batch_sizes, gpu_time_taken, marker="o", label="GPU Time (fp16) bmm")
        plt.plot(batch_sizes, cpu_time_taken, marker="x", label="CPU Time (fp16) plus")
        plt.xlabel("Batch Size")
        plt.ylabel("Time Taken (seconds)")
        plt.title("Matrix Multiplication Time: GPU vs CPU (With Reused Matrices)")
        plt.grid(True)
        plt.legend()
        plt.savefig("mm_reused_comparison.png")


class SecureBartModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.ops.load_library("libSecureRAGExtension.so")
        cls.SecureRAGExtension = torch.ops.SecureRAGExtension
        cls.SecureRAGExtension.openSGX()

    def test_copytensortosgx(self):
        tensor = torch.rand(1024).view(256, 4).to(torch.float32)
        tensorRef = self.SecureRAGExtension.copyTensorToSGX(tensor)
        tensorFromSGX: torch.Tensor = self.SecureRAGExtension.copyTensorFromSGX(tensorRef)
        self.assertTrue(torch.allclose(tensor, tensorFromSGX, atol=1e-4))

    def test_linear(self):
        N = 1024
        indim = 1024
        outdim = 512
        linear = torch.nn.Linear(indim, outdim, True, dtype=torch.float32)
        a = torch.rand((N, indim), dtype=torch.float32)
        b = linear.weight
        c = linear.bias
        out = self.SecureRAGExtension.secureLinear(a, b, c)
        ans = F.linear(a, b, c)
        self.assertTrue(torch.allclose(out, ans, atol=1e-4))

    def test_attention(self):
        embedDim = 768
        heads = 4
        attn = Attention(embed_dim=embedDim, num_heads=heads)
        bsz = 8
        tgtlen = 4
        input = torch.rand(embedDim * bsz * tgtlen).view(tgtlen, bsz, embedDim)
        key = input
        output = attn(input, key)


if __name__ == "__main__":
    unittest.main()
