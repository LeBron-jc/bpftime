import torch
import time

a = torch.randn(1024, 1024, device='cuda')
b = torch.randn(1024, 1024, device='cuda')

for _ in range(3):
    c = torch.mm(a, b)
    torch.cuda.synchronize()
    d = torch.relu(c)
    torch.cuda.synchronize()
    time.sleep(0.5)

print("done")
