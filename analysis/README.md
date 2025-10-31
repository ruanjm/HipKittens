
## Instructions to reproduce benchmarking

Clone:
```bash
git clone https://github.com/HazyResearch/AMD-benchmarking-harness/
cd AMD-benchmarking-harness/ThunderKittens-HIP
git checkout port/
source env.src
```

Get the right docker:
```
podman pull docker.io/rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta

podman run -it \
    --ipc=host \
    --network=host \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    -v $(pwd):/workdir/ \
    -e USE_FASTSAFETENSOR=1 \
    -e SAFETENSORS_FAST_GPU=1 \
    rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta \
    bash
```

Run benchmarking:

BF16 GEMM benchmarking:
```bash
cd https://github.com/HazyResearch/AMD-benchmarking-harness/tree/main/analysis/bf16_gemm/mi350x
bash mi355x_benchmark.sh
```

Rotary benchmarking:
```bash
cd https://github.com/HazyResearch/AMD-benchmarking-harness/tree/main/analysis/rotary/mi350x
bash mi355x_benchmark.sh
```

Layernorm benchmarking:
```bash
cd https://github.com/HazyResearch/AMD-benchmarking-harness/tree/main/analysis/layernorm/mi350x
bash mi355x_benchmark.sh
```

Attention forwards benchmarking:
```bash
# First switch dockers
podman pull docker.io/rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_alpha
podman run -it \
    --ipc=host \
    --network=host \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    -v $(pwd):/workdir/ \
    -e USE_FASTSAFETENSOR=1 \
    -e SAFETENSORS_FAST_GPU=1 \
    rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_alpha \
    bash

# Then run
cd https://github.com/HazyResearch/AMD-benchmarking-harness/tree/main/analysis/attn/fwd/mi350x
bash mi355x_benchmark.sh
```

FP8 GEMM:
```bash
# First switch ThunderKittens branches
cd AMD-benchmarking-harness/ThunderKittens-HIP/
git checkout drew/fp8-4-warps-16x16x128/
source env.src
cd ThunderKittens-HIP/kernels/matmul/FP8_4wave/

# Set the M, N, K in the matmul.cu file
make clean && make
./matmul
```


Each benchmarking script produces a file like ```mi355x...json```. You can plot this with the plot.py script under each kernel folder. 


Common issues:
- If you see a complaint that AITER is not building in the test_python.py files, then instal AITER from source [following this README.md](https://github.com/ROCm/aiter/tree/main)
- If you see an error that ```bin/hipcc/``` is not found, then edit the Makefile to replace ROCM_BUILD_DIR with ```/opt/rocm/bin/hipcc```


## Baselines

We compare to:
- Composable Kernel
- AITER
- PyTorch
- HIPBLASLT
- Triton

To see how we produced the baseline method results, please see [analysis/baselines/README.md](https://github.com/HazyResearch/AMD-benchmarking-harness/tree/main/analysis/baselines).



