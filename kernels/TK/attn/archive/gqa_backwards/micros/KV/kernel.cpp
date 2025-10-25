#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int ATTN_D = 128; // dimension
constexpr int BLOCK_SIZE_KV = 64; // block size for KV

template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using kv_tile = rt<T, BLOCK_SIZE_KV, D, L, S>;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;

template<int D> struct micro_globals {
    gl<bf16, -1, -1, -1, -1> in;
    gl<bf16, -1, -1, -1, -1> out;
    dim3 grid()  { return dim3(1); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int D> __global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals<D> g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE_KV, D, st_8x32_s> (&A_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, D, st_8x32_s>>();

    // Register tiles
    kv_tile<D> K_reg;
    kv_tile<D, bf16, col_l, rt_32x32_s> V_reg;

    const int warpid = kittens::warpid();

    // Load KV data using the KV head index
    G::load(A_smem, g.in, {0, 0, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Load K_j from SMEM to registers  
    load(V_reg, A_smem);
    // load(K_reg, A_smem);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Store K_j to output
    store(g.out, V_reg, {0, 0, 0, 0});
    // store(g.out, K_reg, {0, 0, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
}

template<int D>
void dispatch_micro(micro_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro<ATTN_D>>(m, "dispatch_micro", &micro_globals<ATTN_D>::in, &micro_globals<ATTN_D>::out);
}

