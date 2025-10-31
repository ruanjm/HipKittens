import torch
import random
import time
import sys
import subprocess
import os

torch.manual_seed(0)
random.seed(0)

# Inputs
N = 8192
A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0  
B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0  
Bt = B.t().contiguous()  # Transpose B for the kernel


num_warmup = 500
num_iters = 100

start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = (2 * N**3)  # FLOPs for reference

# Reference matmul using PyTorch
for _ in range(num_warmup):
    C_pytorch = torch.matmul(A, Bt)
timings_pytorch = []
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    C_pytorch = torch.matmul(A, Bt)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings_pytorch.append(elapsed_time)
print(f"{C_pytorch.dtype=}")
avg_time_pytorch = sum(timings_pytorch) / len(timings_pytorch)
tflops_pytorch = flops_ref / (avg_time_pytorch * 1e9) 
print(f"PyTorch reference average execution time: {avg_time_pytorch:.4f} ms")
print(f"PyTorch reference performance: {tflops_pytorch:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")






