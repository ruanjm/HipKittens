#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

#define STR2(x) #x
#define STR(x)  STR2(x)

__shared__ __hip_bfloat16 shared_mem[2048];

#define DS_READ_B32_FUNC(VGPR) \
__device__ __forceinline__ void ds_read_b32_##VGPR(int offset) { \
  const uint32_t smem_ptr = reinterpret_cast<uintptr_t>(&shared_mem[offset]); \
  asm volatile("ds_read_b32 v" STR(VGPR) ", %0 offset:0" \
               :: "v"(smem_ptr) \
               : "memory", "v" STR(VGPR)); \
}

#define DS_READ_B64_FUNC(VGPR0, VGPR1) \
__device__ __forceinline__ void ds_read_b64_##VGPR0##_##VGPR1(int offset) { \
  const uint32_t smem_ptr = reinterpret_cast<uintptr_t>(&shared_mem[offset]); \
  asm volatile("ds_read_b64 v[" STR(VGPR0) ":" STR(VGPR1) "], %0 offset:0" \
               :: "v"(smem_ptr) \
               : "memory", "v" STR(VGPR0), "v" STR(VGPR1)); \
}

#define DS_READ_B128_FUNC(VGPR0, VGPR1, VGPR2, VGPR3) \
__device__ __forceinline__ void ds_read_b128_##VGPR0##_##VGPR1##_##VGPR2##_##VGPR3(int offset) { \
  const uint32_t smem_ptr = reinterpret_cast<uintptr_t>(&shared_mem[offset]); \
  asm volatile("ds_read_b128 v[" STR(VGPR0) ":" STR(VGPR3) "], %0 offset:0" \
               :: "v"(smem_ptr) \
               : "memory", "v" STR(VGPR0), "v" STR(VGPR1), "v" STR(VGPR2), "v" STR(VGPR3)); \
}

#define DS_READ_B128_AGPR_FUNC(AGPR0, AGPR1, AGPR2, AGPR3) \
__device__ __forceinline__ void ds_read_b128_agpr_##AGPR0##_##AGPR1##_##AGPR2##_##AGPR3(int offset) { \
  const uint32_t smem_ptr = reinterpret_cast<uintptr_t>(&shared_mem[offset]); \
  asm volatile("ds_read_b128 a[" STR(AGPR0) ":" STR(AGPR3) "], %0 offset:0" \
               :: "v"(smem_ptr) \
               : "memory", "a" STR(AGPR0), "a" STR(AGPR1), "a" STR(AGPR2), "a" STR(AGPR3)); \
}

#define READ_B64_FUNC(VGPR0, VGPR1) \
__device__ __forceinline__ void v_mov_b64_##VGPR0##_##VGPR1(uint32_t& result1, uint32_t& result2) { \
  asm volatile("v_mov_b32 %0, v" STR(VGPR0) : "=v"(result1) :: "v" STR(VGPR0)); \
  asm volatile("v_mov_b32 %0, v" STR(VGPR1) : "=v"(result2) :: "v" STR(VGPR1)); \
}

#define READ_B128_FUNC(VGPR0, VGPR1, VGPR2, VGPR3) \
__device__ __forceinline__ void v_mov_b128_##VGPR0##_##VGPR1##_##VGPR2##_##VGPR3(uint32_t& result1, uint32_t& result2, uint32_t& result3, uint32_t& result4) { \
  asm volatile("v_mov_b32 %0, v" STR(VGPR0) : "=v"(result1) :: "v" STR(VGPR0)); \
  asm volatile("v_mov_b32 %0, v" STR(VGPR1) : "=v"(result2) :: "v" STR(VGPR1)); \
  asm volatile("v_mov_b32 %0, v" STR(VGPR2) : "=v"(result3) :: "v" STR(VGPR2)); \
  asm volatile("v_mov_b32 %0, v" STR(VGPR3) : "=v"(result4) :: "v" STR(VGPR3)); \
}

#define ACCVGPR_READ_B128_FUNC(AGPR0, AGPR1, AGPR2, AGPR3) \
__device__ __forceinline__ void accvgpr_read_b128_##AGPR0##_##AGPR1##_##AGPR2##_##AGPR3(uint32_t& result1, uint32_t& result2, uint32_t& result3, uint32_t& result4) { \
  asm volatile("v_accvgpr_read_b32 %0, a" STR(AGPR0) : "=v"(result1) :: "a" STR(AGPR0)); \
  asm volatile("v_accvgpr_read_b32 %0, a" STR(AGPR1) : "=v"(result2) :: "a" STR(AGPR1)); \
  asm volatile("v_accvgpr_read_b32 %0, a" STR(AGPR2) : "=v"(result3) :: "a" STR(AGPR2)); \
  asm volatile("v_accvgpr_read_b32 %0, a" STR(AGPR3) : "=v"(result4) :: "a" STR(AGPR3)); \
}

DS_READ_B128_FUNC(10, 11, 12, 13)
DS_READ_B128_FUNC(20, 21, 22, 23)
DS_READ_B128_FUNC(30, 31, 32, 33)
DS_READ_B128_FUNC(40, 41, 42, 43)

DS_READ_B128_AGPR_FUNC(10, 11, 12, 13)
DS_READ_B128_AGPR_FUNC(20, 21, 22, 23)
DS_READ_B128_AGPR_FUNC(30, 31, 32, 33)
DS_READ_B128_AGPR_FUNC(40, 41, 42, 43)

DS_READ_B64_FUNC(10, 11)
DS_READ_B64_FUNC(20, 21)
DS_READ_B64_FUNC(30, 31)
DS_READ_B64_FUNC(40, 41)

READ_B128_FUNC(10, 11, 12, 13)
READ_B128_FUNC(20, 21, 22, 23)
READ_B128_FUNC(30, 31, 32, 33)
READ_B128_FUNC(40, 41, 42, 43)

READ_B64_FUNC(10, 11)
READ_B64_FUNC(20, 21)
READ_B64_FUNC(30, 31)
READ_B64_FUNC(40, 41)

ACCVGPR_READ_B128_FUNC(10, 11, 12, 13)
ACCVGPR_READ_B128_FUNC(20, 21, 22, 23)
ACCVGPR_READ_B128_FUNC(30, 31, 32, 33)
ACCVGPR_READ_B128_FUNC(40, 41, 42, 43)

template<int VGPR_START>
__device__ __forceinline__ void ds_read_b64(int offset) {
  if constexpr (VGPR_START == 10) {
    ds_read_b64_10_11(offset);
  } else if constexpr (VGPR_START == 20) {
    ds_read_b64_20_21(offset);
  } else if constexpr (VGPR_START == 30) {
    ds_read_b64_30_31(offset);
  } else if constexpr (VGPR_START == 40) {
    ds_read_b64_40_41(offset);
  }
}

template<int VGPR_START>
__device__ __forceinline__ void ds_read_b128(int offset) {
  if constexpr (VGPR_START == 10) {
    ds_read_b128_10_11_12_13(offset);
  } else if constexpr (VGPR_START == 20) {
    ds_read_b128_20_21_22_23(offset);
  } else if constexpr (VGPR_START == 30) {
    ds_read_b128_30_31_32_33(offset);
  } else if constexpr (VGPR_START == 40) {
    ds_read_b128_40_41_42_43(offset);
  }
}

template<int AGPR_START>
__device__ __forceinline__ void ds_read_b128_agpr(int offset) {
  if constexpr (AGPR_START == 10) {
    ds_read_b128_agpr_10_11_12_13(offset);
  } else if constexpr (AGPR_START == 20) {
    ds_read_b128_agpr_20_21_22_23(offset);
  } else if constexpr (AGPR_START == 30) {
    ds_read_b128_agpr_30_31_32_33(offset);
  } else if constexpr (AGPR_START == 40) {
    ds_read_b128_agpr_40_41_42_43(offset);
  }
}

template<int VGPR_START>
__device__ __forceinline__ void v_mov_b64(uint32_t& result1, uint32_t& result2) {
  if constexpr (VGPR_START == 10) {
    v_mov_b64_10_11(result1, result2);
  } else if constexpr (VGPR_START == 20) {
    v_mov_b64_20_21(result1, result2);
  } else if constexpr (VGPR_START == 30) {
    v_mov_b64_30_31(result1, result2);
  } else if constexpr (VGPR_START == 40) {
    v_mov_b64_40_41(result1, result2);
  }
}

template<int VGPR_START>
__device__ __forceinline__ void v_mov_b128(uint32_t& result1, uint32_t& result2, uint32_t& result3, uint32_t& result4) {
  if constexpr (VGPR_START == 10) {
    v_mov_b128_10_11_12_13(result1, result2, result3, result4);
  } else if constexpr (VGPR_START == 20) {
    v_mov_b128_20_21_22_23(result1, result2, result3, result4);
  } else if constexpr (VGPR_START == 30) {
    v_mov_b128_30_31_32_33(result1, result2, result3, result4);
  } else if constexpr (VGPR_START == 40) {
    v_mov_b128_40_41_42_43(result1, result2, result3, result4);
  }
}

template<int AGPR_START>
__device__ __forceinline__ void accvgpr_read_b128(uint32_t& result1, uint32_t& result2, uint32_t& result3, uint32_t& result4) {
  if constexpr (AGPR_START == 10) {
    accvgpr_read_b128_10_11_12_13(result1, result2, result3, result4);
  } else if constexpr (AGPR_START == 20) {
    accvgpr_read_b128_20_21_22_23(result1, result2, result3, result4);
  } else if constexpr (AGPR_START == 30) {
    accvgpr_read_b128_30_31_32_33(result1, result2, result3, result4);
  } else if constexpr (AGPR_START == 40) {
    accvgpr_read_b128_40_41_42_43(result1, result2, result3, result4);
  }
}

template<int VGPR_ID>
__device__ __forceinline__ void process_via_shared_memory(const __hip_bfloat16* input_data, __hip_bfloat16* output_ptr, int tid) {
  // Each thread writes two input values to shared memory
  for (int i = 0; i < 8; i++) {
    shared_mem[threadIdx.x * 8 + i] = input_data[tid * 8 + i];
  }
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();

  // Read two floats from shared memory to VGPR pair using ds_read_b64
  // ds_read_b128<VGPR_ID>(threadIdx.x * 8);
  ds_read_b128_agpr<VGPR_ID>(threadIdx.x * 8);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();

  uint32_t stored_value1, stored_value2, stored_value3, stored_value4;
  // v_mov_b128<VGPR_ID>(stored_value1, stored_value2, stored_value3, stored_value4);
  accvgpr_read_b128<VGPR_ID>(stored_value1, stored_value2, stored_value3, stored_value4);

  __hip_bfloat162 packed_value1 = *reinterpret_cast<__hip_bfloat162*>(&stored_value1);
  __hip_bfloat162 packed_value2 = *reinterpret_cast<__hip_bfloat162*>(&stored_value2);
  __hip_bfloat162 packed_value3 = *reinterpret_cast<__hip_bfloat162*>(&stored_value3);
  __hip_bfloat162 packed_value4 = *reinterpret_cast<__hip_bfloat162*>(&stored_value4);

  float packed_value1_x = __bfloat162float(packed_value1.x);
  float packed_value1_y = __bfloat162float(packed_value1.y);
  float packed_value2_x = __bfloat162float(packed_value2.x);
  float packed_value2_y = __bfloat162float(packed_value2.y);
  float packed_value3_x = __bfloat162float(packed_value3.x);
  float packed_value3_y = __bfloat162float(packed_value3.y);
  float packed_value4_x = __bfloat162float(packed_value4.x);
  float packed_value4_y = __bfloat162float(packed_value4.y);
  packed_value1.x = __float2bfloat16(packed_value1_x * 2.0f + 1.0f);
  packed_value1.y = __float2bfloat16(packed_value1_y * 2.0f + 1.0f);
  packed_value2.x = __float2bfloat16(packed_value2_x * 2.0f + 1.0f);
  packed_value2.y = __float2bfloat16(packed_value2_y * 2.0f + 1.0f);
  packed_value3.x = __float2bfloat16(packed_value3_x * 2.0f + 1.0f);
  packed_value3.y = __float2bfloat16(packed_value3_y * 2.0f + 1.0f);
  packed_value4.x = __float2bfloat16(packed_value4_x * 2.0f + 1.0f);
  packed_value4.y = __float2bfloat16(packed_value4_y * 2.0f + 1.0f);


  output_ptr[0] = packed_value1.x;
  output_ptr[1] = packed_value1.y;
  output_ptr[2] = packed_value2.x;
  output_ptr[3] = packed_value2.y;
  output_ptr[4] = packed_value3.x;
  output_ptr[5] = packed_value3.y;
  output_ptr[6] = packed_value4.x;
  output_ptr[7] = packed_value4.y;
}

extern "C" __global__
void simple_load_store_kernel(const __hip_bfloat16* __restrict__ input_data,
                              __hip_bfloat16* __restrict__ output_data,
                              int num_elements) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  process_via_shared_memory<10>(input_data, &output_data[tid * 8], tid);
}

