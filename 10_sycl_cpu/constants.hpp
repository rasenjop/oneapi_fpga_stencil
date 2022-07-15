#include <vector>
#include <chrono>
#include <iostream>

#include <oneapi/tbb/cache_aligned_allocator.h>

// the array size of input and output matrix
constexpr size_t kRows= 1024*32;
constexpr size_t kCols= 1024;
constexpr size_t kArraySize = kRows * kCols;

using FloatVector = std::vector<float,oneapi::tbb::cache_aligned_allocator<float>>; 

void run_fpga_kernel(FloatVector& in, FloatVector& m, FloatVector& out);
