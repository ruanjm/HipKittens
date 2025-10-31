
## Setup mi350x

Pull docker:
```
docker pull rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_prealpha
```

Launch docker:
```
sudo docker run -it \
    --ipc=host \
    --network=host \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    -v $(pwd):/workdir \
    -e USE_FASTSAFETENSOR=1 \
    -e SAFETENSORS_FAST_GPU=1 \
    rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_prealpha \
    bash
```

Launch MI355X docker:
```
podman pull docker.io/rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta
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
    rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta \
    bash
```
```
salloc --exclusive --mem=0
```

## Failure modes

If all the files become root-owned, run this command to fix it (for user id 23120 and guest id 100):
```bash
sudo chown -R 1003:1003 /workdir/
```

cd /workdir/AMD-benchmarking-harness/ThunderKittens-HIP/tests/unit/
cd /workdir/AMD-benchmarking-harness/kernels/TK/micro/cdna3/vec_red/
make clean && make -j64 > out.log 2>&1
rm -rf outputs/* && ./unit_tests outputs > out.log 2>&1

