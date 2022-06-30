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
sycl::event RunKernel(queue& q, buffer<float,1>& b_input, buffer<float,1>& b_mask,
               buffer<float,1>& b_output){
    // submit the kernel
    auto e = q.submit([&](handler &h) {
      //Properties for HBM
      ext::oneapi::accessor_property_list HBM0{ext::intel::buffer_location<0>};
      ext::oneapi::accessor_property_list HBM0_noInit{no_init,ext::intel::buffer_location<0>};
      // Data accessors
      accessor input{b_input, h, read_only, HBM0};
      accessor mask{b_mask, h, read_only};
      accessor output{b_output, h, write_only, HBM0_noInit};

      // Kernel executes with pipeline parallelism on the FPGA.
      // Use kernel_args_restrict to specify that input, mask, and output do not alias.
      h.single_task<Stencil>([=]() [[intel::kernel_args_restrict]] {
          float local_mask[9];
          for(int i=0; i<9; i++) local_mask[i]=mask[i];
          for(int i=1; i<kRows-1; i++){
            int crow_base=i*kCols;
            int prow_base = crow_base - kCols;
            int nrow_base = crow_base + kCols;
            for(int j=1; j<kCols-1; j++){
              float tmp = local_mask[0] * input[prow_base + j - 1] + 
                          local_mask[1] * input[prow_base + j    ] +
                          local_mask[2] * input[prow_base + j + 1] +
                          local_mask[3] * input[crow_base + j - 1] +
                          local_mask[4] * input[crow_base + j    ] +
                          local_mask[5] * input[crow_base + j + 1] +
                          local_mask[6] * input[nrow_base + j - 1] +
                          local_mask[7] * input[nrow_base + j    ] +
                          local_mask[8] * input[nrow_base + j + 1] ;
              output[crow_base+j] = tmp;
            }
          }
      });
    });
    return e;
}
