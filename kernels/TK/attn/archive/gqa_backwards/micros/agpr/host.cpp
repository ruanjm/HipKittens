#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <iostream>
#include <vector>
#include <chrono>

extern "C" __global__
void simple_load_store_kernel(const hip_bfloat16* __restrict__ input_data,
                              hip_bfloat16* __restrict__ output_data,
                              int num_elements);

#define HIP_CHECK(call) \
  do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(err) << std::endl; \
      exit(1); \
    } \
  } while(0)

int main() {
  const int num_elements = 2048;
  const size_t size = num_elements * sizeof(hip_bfloat16);

  std::vector<hip_bfloat16> h_input(num_elements);
  std::vector<hip_bfloat16> h_output(num_elements);

  for (int i = 0; i < num_elements; i++) {
    h_input[i] = ((hip_bfloat16)(static_cast<float>(i) * 0.1f));
  }

  hip_bfloat16 *d_input, *d_output;
  HIP_CHECK(hipMalloc(&d_input, size));
  HIP_CHECK(hipMalloc(&d_output, size));

  HIP_CHECK(hipMemcpy(d_input, h_input.data(), size, hipMemcpyHostToDevice));

  int elements_per_thread = 8;
  dim3 blockSize(256);
  dim3 gridSize(((num_elements / elements_per_thread) + blockSize.x - 1) / blockSize.x);

  auto start = std::chrono::high_resolution_clock::now();

  hipLaunchKernelGGL(simple_load_store_kernel, gridSize, blockSize, 0, 0,
                     d_input, d_output, num_elements);

  HIP_CHECK(hipDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  HIP_CHECK(hipMemcpy(h_output.data(), d_output, size, hipMemcpyDeviceToHost));

  std::cout << "Kernel execution time: " << duration.count() << " microseconds" << std::endl;

  std::cout << "First 10 results:" << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << "input[" << i << "] = " << h_input[i]
              << " -> output[" << i << "] = " << h_output[i] << std::endl;
  }

  bool success = true;
  for (int i = 0; i < num_elements; i++) {
    // Convert input to float, do computation, convert back to bf16, then to float
    float input_as_float = static_cast<float>(h_input[i]);
    hip_bfloat16 expected_bf16 = static_cast<hip_bfloat16>(input_as_float * 2.0f + 1.0f);
    float expected = static_cast<float>(expected_bf16);
    float actual = static_cast<float>(h_output[i]);

    if (abs(actual - expected) > 1e-6) {
      std::cout << "Mismatch at index " << i << ": expected " << expected
                << ", got " << actual << std::endl;
      success = false;
      break;
    }
  }

  if (success) {
    std::cout << "All results match expected values!" << std::endl;
  }

  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_output));

  return 0;
}