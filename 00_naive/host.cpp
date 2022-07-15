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

// the tolerance used in floating point comparisons
constexpr float kTol = 0.001;

void gold_stencil(const std::vector<float>& input, float (&mask)[3][3], std::vector<float>& res){

  for(int i=1; i<kRows-1; i++){
    for(int j=1; j<kCols-1; j++){
      res[i*kCols+j]=0;
      for(int m=0; m<3; m++){
        int row=i+m-1;
        for (int n=0; n<3; n++){
          res[i*kCols+j]+= mask[m][n] * input[row*kCols + j + n - 1];
        }
      }
    }
  }
}

int main() {
  std::vector<float> input(kArraySize);
  std::vector<float> output(kArraySize);
  std::vector<float> mask2{2,4,2,4,1,4,2,4,2};
  float mask[3][3]={2,4,2,4,1,4,2,4,2};

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
    queue q(device_selector, dpc_common::exception_handler);

    // create the device buffers
    buffer d_input{input};
    buffer d_mask{mask2};//, range<1>{9}};
    buffer d_output{output};

    // The definition of this function is in a different compilation unit,
    // so host and device code can be separately compiled.
    auto start = std::chrono::high_resolution_clock::now();
    RunKernel(q, d_input, d_mask, d_output);
    q.wait();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time FPGA: "<< std::chrono::duration<double,std::milli>(end - start).count() << " ms.\n";

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
  std::vector<float> gold_output(kArraySize); 
  auto start = std::chrono::high_resolution_clock::now();
  gold_stencil(input, mask, gold_output);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Time CPU: "<< std::chrono::duration<double,std::milli>(end - start).count() << " ms.\n";
  size_t correct = 0;
  for (size_t i = 0; i < kArraySize; i++) {
    float tmp = gold_output[i] - output[i];
    if (tmp * tmp < kTol * kTol) {
      correct++;
    }
  }

  // Summarize results
  if (correct >= (kArraySize - 2*kRows -2*kCols +4)) { //border is not considered
    std::cout << "PASSED: results are correct\n";
  } else {
    std::cout << "FAILED: results are incorrect\n";
  }
  
  return !(correct >= (kArraySize - 2*kRows -2*kCols +4));
}