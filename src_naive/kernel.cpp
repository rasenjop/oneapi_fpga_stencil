//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "kernel.hpp"


// This file contains 'almost' exclusively device code. The single-source SYCL
// code has been refactored between host.cpp and kernel.cpp to separate host and
// device code to the extent that the language permits.
//
// Note that ANY change in either this file or in kernel.hpp will be detected
// by the build system as a difference in the dependencies of device.o,
// triggering a full recompilation of the device code. 
//
// This is true even of trivial changes, e.g. tweaking the function definition 
// or the names of variables like 'q' or 'h', EVEN THOUGH these are not truly 
// "device code".


// Forward declare the kernel names in the global scope. This FPGA best practice
// reduces compiler name mangling in the optimization reports.
class Stencil;
//RunKernel(q, d_input, d_mask, d_output);
void RunKernel(queue& q, buffer<float,1>& b_input, buffer<float,1>& b_mask,
               buffer<float,1>& b_output){
    // submit the kernel
    q.submit([&](handler &h) {
      // Data accessors
      accessor input{b_input, h, read_only};
      accessor mask{b_mask, h, read_only};
      accessor output{b_output, h, write_only, no_init};

      // Kernel executes with pipeline parallelism on the FPGA.
      // Use kernel_args_restrict to specify that input, mask, and output do not alias.
      h.single_task<Stencil>([=]() [[intel::kernel_args_restrict]] {
          for(int i=1; i<kRows-1; i++){
            for(int j=1; j<kCols-1; j++){
              output[i*kCols+j]=0;
              for(int m=0; m<3; m++){
                int row=i+m-1;
                for (int n=0; n<3; n++){
                  output[i*kCols+j]+= mask[m*3+n] * input[row*kCols + j + n - 1];
                }
              }
            }
          }
      });
    });
}
