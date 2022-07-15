//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>

#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/parallel_for.h>
#include "constants.hpp"

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

void parallel_stencil(const FloatVector& input, const FloatVector& mask, FloatVector& res){

  tbb::parallel_for(1UL, static_cast<unsigned long>(kRows-1), [&](int i){
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
  });
}

int main() {
  FloatVector input(kArraySize); 
  FloatVector output(kArraySize);
  FloatVector mask{2,4,2,4,1,4,2,4,2};

  // Fill input with random float values
  for (size_t i = 0; i < kArraySize; i++) {
    input[i] = rand() / (float)RAND_MAX;
  }

  run_fpga_kernel(input, mask, output);
  run_cpu_kernel(input, mask, output);

  // Test the results
  FloatVector gold_output(kArraySize); 
  auto start = std::chrono::high_resolution_clock::now();
  //gold_stencil(input, mask, gold_output);
  parallel_stencil(input, mask, gold_output);
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
