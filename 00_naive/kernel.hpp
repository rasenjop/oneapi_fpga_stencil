//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

using namespace sycl;
// CL_DEVICE_MAX_MEM_ALLOC_SIZE=info::device::max_mem_alloc_size=536869888 bytes = 127 MBytes
// the array size of input and output matrix
constexpr size_t kRows=1024*10;
constexpr size_t kCols=1024;
constexpr size_t kArraySize = kRows * kCols;

void RunKernel(queue& q, buffer<float,1>& b_input, buffer<float,1>& b_mask,
               buffer<float,1>& b_output);
