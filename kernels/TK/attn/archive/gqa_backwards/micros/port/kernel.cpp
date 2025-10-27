#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;

struct micro_globals {
  gl<bf16, -1, -1, -1, -1> A;
  gl<bf16, -1, -1, -1, -1> A_tk;
  dim3 grid() {return dim3(1); }
  dim3 block() {return dim3(NUM_THREADS);}
  size_t dynamic_shared_memory() {return MAX_SHARED_MEMORY;}
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int*)&__shm[0]);
  st<bf16, 32, 32, st_16x16_s> (&A_smem) = al.allocate<st<bf16, 32, 32, st_16x16_s>>();

  rt_bf<32, 32, col_l, rt_16x32_s> A_reg;

  load(A_smem, g.A, {0, 0, 0, 0});
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();

  load(A_reg, A_smem);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();

  store(g.A_tk, A_reg, {0, 0, 0, 0});
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
}

void dispatch_micro(micro_globals g) {
  unsigned long mem_size = g.dynamic_shared_memory();
  hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
  micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
  hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
  py::bind_function<dispatch_micro>(m, "dispatch_micro", 
    &micro_globals::A, 
    &micro_globals::A_tk
  );
}


