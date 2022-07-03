//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#include <vector>
#include <chrono>

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <oneapi/tbb/cache_aligned_allocator.h>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

// This code sample demonstrates how to split the host and FPGA kernel code into
// separate compilation units so that they can be separately recompiled.
// Consult the README for a detailed discussion.
//  - host.cpp (this file) contains exclusively code that executes on the host.
//  - kernel.cpp contains almost exclusively code that executes on the device.
//  - kernel.hpp contains only the forward declaration of a function containing
//    the device code.
#include "kernel.hpp"

using namespace sycl;
using FloatVector = std::vector<float,oneapi::tbb::cache_aligned_allocator<float>>; 


// the tolerance used in floating point comparisons
constexpr float kTol = 0.001;

void gold_stencil(const FloatVector& input, const FloatVector& mask, FloatVector& res){

  for(int i=1; i<kRows-1; i++){
    int crow_base=i*kCols;
    int prow_base = crow_base - kCols;
    int nrow_base = crow_base + kCols;
    for(int j=1; j<kCols-1; j++){
      float tmp = mask[0] * input[prow_base + j - 1] + 
                  mask[1] * input[prow_base + j    ] +
                  mask[2] * input[prow_base + j + 1] +
                  mask[3] * input[crow_base + j - 1] +
                  mask[4] * input[crow_base + j    ] +
                  mask[5] * input[crow_base + j + 1] +
                  mask[6] * input[nrow_base + j - 1] +
                  mask[7] * input[nrow_base + j    ] +
                  mask[8] * input[nrow_base + j + 1] ;
      res[crow_base+j] = tmp;
    }
  }
}

int main() {
  FloatVector input(kArraySize+3); //+3 due to the shift register loading 3 elems. in advance
  FloatVector output(kArraySize);
  FloatVector mask{2,4,2,4,1,4,2,4,2};

  // Fill input with random float values
  for (size_t i = 0; i < kArraySize; i++) {
    input[i] = rand() / (float)RAND_MAX;
  }

  // Select either the FPGA emulator or FPGA device
#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector device_selector;
#else
  ext::intel::fpga_selector device_selector;
#endif

  try {

    // Create a queue bound to the chosen device.
    // If the device is unavailable, a SYCL runtime exception is thrown.
    auto prop_list = property_list{property::queue::enable_profiling()};
    queue q(device_selector, dpc_common::exception_handler, prop_list);

    // create the device buffers
    buffer d_input{input};
    buffer d_mask{mask};
    buffer d_output{output};

    // The definition of this function is in a different compilation unit,
    // so host and device code can be separately compiled.
    auto start = std::chrono::high_resolution_clock::now();
    auto e = RunKernel(q, d_input, d_mask, d_output);
    q.wait();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time FPGA: "<< std::chrono::duration<double,std::milli>(end - start).count() << " ms.\n";
    ReportTime("FPGA Stencil with HBM. Time: ",e);

 } catch (exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  // At this point, the device buffers have gone out of scope and the kernel
  // has been synchronized. Therefore, the output data (output) has been updated
  // with the results of the kernel and is safely accesible by the host CPU.

  // Test the results
  FloatVector gold_output(kArraySize); 
  auto start = std::chrono::high_resolution_clock::now();
  gold_stencil(input, mask, gold_output);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Time CPU: "<< std::chrono::duration<double,std::milli>(end - start).count() << " ms.\n";
  size_t incorrect = 0;
  for (size_t i = 1; i < kRows-1; i++) {
    for (size_t j = 1; j < kCols-1; j++) {
      float tmp = gold_output[i*kCols+j] - output[i*kCols+j];
      if (tmp * tmp >= kTol * kTol) {
        incorrect++;
        std::cout << "Error at index i=" << i << " j=" << j << " ; gold=" << gold_output[i*kCols+j] << "; out="<< output[i*kCols+j] << '\n';
      }
    }
  }

  // Summarize results
  if (!incorrect) {
    std::cout << "PASSED: results are correct\n";
  } else {
    std::cout << "FAILED: results are incorrect\n";
  }

  return incorrect;
}
