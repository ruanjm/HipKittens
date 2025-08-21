import torch
import random
import math
import aiter

torch.manual_seed(0)
random.seed(0)

torch.set_printoptions(
    precision=3,        
    sci_mode=False,     
    linewidth=220,      
    threshold=float("inf")  
)

def reference_backwards(Q, K, V, dO, causal):
    """Reference implementation using BHND layout (batch, heads, seq, dim)"""
    # Convert to float64 and create new leaf tensors with requires_grad
    q_ = Q.detach().to(torch.float64).requires_grad_(True)
    k_ = K.detach().to(torch.float64).requires_grad_(True)
    v_ = V.detach().to(torch.float64).requires_grad_(True)
    dO_ = dO.to(torch.float64)
    
    # manual pytorch implementation of scaled dot product attention
    QK = torch.matmul(q_, k_.transpose(-2, -1))
    QK /= (q_.size(-1) ** 0.5)
    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v_)
    
    output.backward(dO_)
    
    q_grad = q_.grad
    k_grad = k_.grad
    v_grad = v_.grad
    
    q_grad = q_grad.to(torch.bfloat16)
    k_grad = k_grad.to(torch.bfloat16)
    v_grad = v_grad.to(torch.bfloat16)
    output = output.to(torch.bfloat16)
    
    return output, q_grad, k_grad, v_grad

def simple_flash_backward(Q, K, V, dO, m, l):
    """Simple version that should match PyTorch exactly"""
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)

    # Recompute scores and probabilities with saved m, l
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.exp(S - m.unsqueeze(-1)) / l.unsqueeze(-1)
    O = torch.matmul(P, V)

    # dV
    dV = torch.matmul(P.transpose(-2, -1), dO)

    # softmax backward
    Delta = (dO * O).sum(dim=-1, keepdim=True)                 # (B, N, H, 1)
    dS = P * (torch.matmul(dO, V.transpose(-2, -1)) - Delta)   # (B, N, H, N)

    # chain rule through S = (Q K^T) * scale
    dQ = torch.matmul(dS, K) * scale
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale

    return dQ, dK, dV

# Parameters
causal = False
b = 16
h = 16
n = 2048
d = 128
dtype = torch.bfloat16

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)
    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    return scaled_tensor.contiguous()

def generate_inputs():
    mean = 1 #1e-1
    std = 0.1  # REDUCED from 10 to 0.1 for numerical stability
    
    # Generate in BHND format (batch, heads, seq, dim) for reference
    Q = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    K = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    V = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    dO = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')

    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)

    return Q, K, V, dO

# Generate base inputs in BHND format
Q_bhnd, K_bhnd, V_bhnd, dO_bhnd = generate_inputs()

# Clone for PyTorch reference (keep BHND format)
Q_pytorch = Q_bhnd.clone().detach().requires_grad_(True)
K_pytorch = K_bhnd.clone().detach().requires_grad_(True)
V_pytorch = V_bhnd.clone().detach().requires_grad_(True)
dO_pytorch = dO_bhnd.clone()

# Create leaf tensors for AITER (BNHD format)
Q_aiter = Q_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  # BHND -> BNHD
K_aiter = K_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  # BHND -> BNHD
V_aiter = V_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  # BHND -> BNHD
dO_aiter = dO_bhnd.transpose(1, 2).contiguous()  # BHND -> BNHD

# Create leaf tensors for TK (BNHD format)
Q_tk = Q_bhnd.clone().contiguous().detach().requires_grad_(True)  # BHND -> BNHD
K_tk = K_bhnd.clone().contiguous().detach().requires_grad_(True)  # BHND -> BNHD
V_tk = V_bhnd.clone().contiguous().detach().requires_grad_(True)  # BHND -> BNHD
dO_tk = dO_bhnd.clone().contiguous()  # BHND -> BNHD

print(f"Tensor shapes:")
print(f"PyTorch (BHND): Q={Q_pytorch.shape}, K={K_pytorch.shape}, V={V_pytorch.shape}")
print(f"AITER (BNHD):   Q={Q_aiter.shape}, K={K_aiter.shape}, V={V_aiter.shape}")
print(f"TK (BNHD):      Q={Q_tk.shape}, K={K_tk.shape}, V={V_tk.shape}")

# AITER forward and backward
print("\nRunning AITER...")
out_aiter, softmax_lse = aiter.flash_attn_func(Q_aiter, K_aiter, V_aiter, causal, return_lse=True, deterministic=True)
out_aiter.backward(dO_aiter)
q_grad_aiter_bnhd = Q_aiter.grad
k_grad_aiter_bnhd = K_aiter.grad  
v_grad_aiter_bnhd = V_aiter.grad
out_aiter_bhnd = out_aiter.transpose(1, 2)  # BNHD -> BHND
q_grad_aiter_bhnd = q_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND
k_grad_aiter_bhnd = k_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND
v_grad_aiter_bhnd = v_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND

# PyTorch Reference
print("Running PyTorch reference...")
print(f"Q_pytorch[0, 0, 0, :16]={Q_pytorch[0, 0, 0, :16]}")
out_pytorch, q_grad_pytorch, k_grad_pytorch, v_grad_pytorch = reference_backwards(Q_pytorch, K_pytorch, V_pytorch, dO_pytorch, causal)

# Tiled
print("Running Tiled forward to get m, l...")
print(f"Q_tk[0, 0, 0, :16]={Q_tk[0, 0, 0, :16]}")
QK = torch.matmul(Q_tk.float(), K_tk.transpose(-2, -1).float()) / math.sqrt(d)
m_tk = QK.max(dim=-1, keepdim=True)[0] 
exp_scores = torch.exp(QK - m_tk)  
l_tk = exp_scores.sum(dim=-1, keepdim=True)  
P_tk = exp_scores / l_tk
O_tk = torch.matmul(P_tk, V_tk.float())
m_tk = m_tk.squeeze(-1)
l_tk = l_tk.squeeze(-1)

dQ_tk, dK_tk, dV_tk = simple_flash_backward(Q_tk.float(), K_tk.float(), V_tk.float(), dO_tk.float(), m_tk, l_tk)
out_tk_bhnd = O_tk
q_grad_tk_bhnd = dQ_tk
k_grad_tk_bhnd = dK_tk
v_grad_tk_bhnd = dV_tk
print(f"m_tk.shape={m_tk.shape}")
print(f"l_tk.shape={l_tk.shape}")


# Compare
out_diff = (out_aiter_bhnd - out_pytorch).abs()
q_grad_diff = (q_grad_aiter_bhnd - q_grad_pytorch).abs()
k_grad_diff = (k_grad_aiter_bhnd - k_grad_pytorch).abs()
v_grad_diff = (v_grad_aiter_bhnd - v_grad_pytorch).abs()

# Compare TK with PyTorch
out_tk_diff = (out_tk_bhnd - out_pytorch).abs()
q_grad_tk_diff = (q_grad_tk_bhnd - q_grad_pytorch).abs()
k_grad_tk_diff = (k_grad_tk_bhnd - k_grad_pytorch).abs()
v_grad_tk_diff = (v_grad_tk_bhnd - v_grad_pytorch).abs()

print(f"\nOutput comparison:")
print(f"Output max error: {out_diff.max().item():.6f}")
print(f"Output mean error: {out_diff.mean().item():.6f}")

print(f"\nGradient comparison:")
print(f"Q grad max error: {q_grad_diff.max().item():.6f}")
print(f"K grad max error: {k_grad_diff.max().item():.6f}")
print(f"V grad max error: {v_grad_diff.max().item():.6f}")

print(f"\nTiled vs PyTorch comparison:")
print(f"Output max error: {out_tk_diff.max().item():.6f}")
print(f"Q grad max error: {q_grad_tk_diff.max().item():.6f}")
print(f"K grad max error: {k_grad_tk_diff.max().item():.6f}")
print(f"V grad max error: {v_grad_tk_diff.max().item():.6f}")




