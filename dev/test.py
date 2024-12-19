import torch
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F


@contextmanager
def timer():
    start = time.time()
    yield
    end = time.time()
    print(f"time elapsed: {end - start} sec")


torch.ops.load_library("libSecureRAGExtension.so")
SecureRAGExtension = torch.ops.SecureRAGExtension
SecureRAGExtension.openSGX()
executor = ThreadPoolExecutor(4)

N = 1024
indim = 1024
outdim = 512
linear = torch.nn.Linear(indim, outdim, True, dtype=torch.float32)
a = torch.rand((N, indim), dtype=torch.float32)
b = linear.weight
c = linear.bias
print(a, b, c)
# a = torch.Tensor([[1]]).type(torch.float32)
# b = torch.Tensor([[1]]).type(torch.float32)
# c = torch.Tensor([1]).type(torch.float32)


def work():
    with timer():
        out = SecureRAGExtension.secureLinear(a, b, c)
    ans = F.linear(a, b, c)
    print(f"{out}, {ans}, {torch.allclose(out, ans, atol=1e-4)}")
    # print(ans)
    # print(torch.allclose(out, ans, atol=1e-5))


result = []
for i in range(10):
    result.append(executor.submit(work))

for it in result:
    it.result()
