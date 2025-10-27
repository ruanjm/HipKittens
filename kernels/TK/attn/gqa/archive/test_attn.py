import torch
import tk_kernel
import tk_kernel_asm
import random
import time
import math
from torch.nn.functional import scaled_dot_product_attention
import aiter

torch.set_printoptions(
    precision=3,        # decimals
    sci_mode=False,     # True → always scientific
    linewidth=220,      # characters per line before folding
    threshold=float("inf")  # print every element, no summarising "..."
)

torch.cuda.set_device(2)

# Inputs
B = 1
H = 1
H_KV = 1
N = 1024
D = 128
causal = False
dtype = torch.bfloat16

def flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    """Calculate FLOPs for attention operation."""
    flop = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return flop

def efficiency(flop, time):
    """Calculate efficiency in TFLOPS."""
    flop = flop / 1e12  # convert to TFLOPS
    time = time / 1e3   # convert to seconds
    return flop / time

def robustness_check(ref, pred):
    ref = ref.float()
    pred = pred.float()
    diff = (ref - pred).abs()
    denom = ref.abs().clamp_min(1e-6)
    mask = (diff > (0.001 + 0.05 * denom))
    error_count = mask.sum().item()
    numel = ref.numel()
    rel_error = error_count / numel
    l2_error = (diff.pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), pred.flatten(), dim=0).item()
    return diff, error_count, numel, rel_error, l2_error, cos, mask  

flops_ref = flops(B, N, H, D, causal)

# AITER
torch.manual_seed(0)
random.seed(0)
q = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
k = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
v = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
out_ref, lse_ref = aiter.flash_attn_func(q, k, v, causal=causal, return_lse=True, deterministic=True)
att_ref = torch.matmul(k.squeeze(0).squeeze(-2), q.squeeze(0).squeeze(-2).transpose(-1, -2))

# HK ASM
out_hk_asm = torch.zeros(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
lse_hk_asm = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda', requires_grad=True)
torch.cuda.synchronize()
tk_kernel_asm.dispatch_micro(q, k, v, out_hk_asm, lse_hk_asm)
torch.cuda.synchronize()

# HK
out_hk = torch.zeros(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
lse_hk = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda', requires_grad=True)
torch.cuda.synchronize()
tk_kernel.dispatch_micro(q, k, v, out_hk, lse_hk)

# Compare against reference
num_print = 16
print(f"\n HK ASM vs AITER comparison:")
print("\nO outputs:")
print("HK: ", out_hk[0, :, 0, 0], "Max:", out_hk.max().item())
print("HK ASM: ", out_hk_asm[0, :, 0, 0], "Max:", out_hk_asm.max().item())
print("AITER: ", out_ref[0, :, 0, 0], "Max:", out_ref.max().item())

print("\nLSE outputs:")
print("HK: ", lse_hk[0, 0, 0, :num_print], "Max:", lse_hk.max().item())
print("HK ASM: ", lse_hk_asm[0, 0, 0, :num_print], "Max:", lse_hk_asm.max().item())
print("AITER: ", lse_ref[0, 0, :num_print], "Max:", lse_ref.max().item())

print("\nHK vs HK ASM comparison:")
o_diff, o_err_cnt, o_total, o_rel_error, o_l2_error, o_cos, o_mask = robustness_check(out_hk, out_hk_asm)
print(f"O: max_abs={o_diff.max().item():.6f}, max_rel={o_rel_error:.4f}, "
      f"rel_l2={o_l2_error:.4f}, cos={o_cos:.6f}, "
      f"errors={o_err_cnt}/{o_total} ({100*o_err_cnt/o_total:.4f}%)")
l_diff, l_err_cnt, l_total, l_rel_error, l_l2_error, l_cos, l_mask = robustness_check(lse_hk, lse_hk_asm)
print(f"LSE: max_abs={l_diff.max().item():.6f}, max_rel={l_rel_error:.4f}, "
      f"rel_l2={l_l2_error:.4f}, cos={l_cos:.6f}, "
      f"errors={l_err_cnt}/{l_total} ({100*l_err_cnt/l_total:.4f}%)")

print("HK vs AITER comparison:")
o_diff, o_err_cnt, o_total, o_rel_error, o_l2_error, o_cos, o_mask = robustness_check(out_hk, out_ref)
print(f"O: max_abs={o_diff.max().item():.6f}, max_rel={o_rel_error:.4f}, "
      f"rel_l2={o_l2_error:.4f}, cos={o_cos:.6f}, "
      f"errors={o_err_cnt}/{o_total} ({100*o_err_cnt/o_total:.4f}%)")
l_diff, l_err_cnt, l_total, l_rel_error, l_l2_error, l_cos, l_mask = robustness_check(lse_hk, lse_ref.unsqueeze(-1).transpose(-1, -2))
print(f"LSE: max_abs={l_diff.max().item():.6f}, max_rel={l_rel_error:.4f}, "
      f"rel_l2={l_l2_error:.4f}, cos={l_cos:.6f}, "
      f"errors={l_err_cnt}/{l_total} ({100*l_err_cnt/l_total:.4f}%)")

print("HK ASM vs AITER comparison:")
o_diff, o_err_cnt, o_total, o_rel_error, o_l2_error, o_cos, o_mask = robustness_check(out_hk_asm, out_ref)
print(f"O: max_abs={o_diff.max().item():.6f}, max_rel={o_rel_error:.4f}, "
      f"rel_l2={o_l2_error:.4f}, cos={o_cos:.6f}, "
      f"errors={o_err_cnt}/{o_total} ({100*o_err_cnt/o_total:.4f}%)")
l_diff, l_err_cnt, l_total, l_rel_error, l_l2_error, l_cos, l_mask = robustness_check(lse_hk_asm, lse_ref.unsqueeze(-1).transpose(-1, -2))
print(f"LSE: max_abs={l_diff.max().item():.6f}, max_rel={l_rel_error:.4f}, "
      f"rel_l2={l_l2_error:.4f}, cos={l_cos:.6f}, "
      f"errors={l_err_cnt}/{l_total} ({100*l_err_cnt/l_total:.4f}%)")
