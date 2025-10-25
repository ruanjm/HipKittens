import torch
import random
import math
import tk_kernel

torch.set_printoptions(
    precision=3,
    sci_mode=False,
    linewidth=220,
    threshold=float("inf")
)

random.seed(0)
torch.manual_seed(0)

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

A = torch.randn((1, 1, 32, 32), dtype=torch.bfloat16, device='cuda')

# Reference
A_ref = A

# ThunderKittens
A_tk = torch.zeros_like(A_ref)

torch.cuda.synchronize()
tk_kernel.dispatch_micro(A, A_tk)
torch.cuda.synchronize()

# check
num_print = 8
print("A_tk: ", A_tk[0, 0, :num_print, :num_print], "Max:", A_tk.max().item())
print("A_ref: ", A_ref[0, 0, :num_print, :num_print], "Max:", A_ref.max().item())

A_diff, A_err_cnt, A_total, A_rel_error, A_l2_error, A_cos, A_mask = robustness_check(A_ref, A_tk)
print(f"A: max_abs={A_diff.max().item():.6f}, max_rel={A_rel_error:.4f}, "
      f"rel_l2={A_l2_error:.4f}, cos={A_cos:.6f}, "
      f"errors={A_err_cnt}/{A_total} ({100*A_err_cnt/A_total:.4f}%)")