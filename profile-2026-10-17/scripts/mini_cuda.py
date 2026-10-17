import torch
a = torch.randn(2048, 2048, device="cuda")
for _ in range(5):
    b = a @ a
torch.cuda.synchronize()
print("mini ok", float(b.sum()))
