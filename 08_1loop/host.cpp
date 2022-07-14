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
#include "unrolled_loop.hpp"
#include "shift_reg.hpp"


using namespace sycl;

// the array size of input and output matrix
constexpr size_t kRows= 1024*32;
constexpr size_t kCols= 1024*32;
constexpr size_t kArraySize = kRows * kCols;

static void ReportTime(const std::string &msg, int k, event e) {
  cl_ulong time_start =
      e.get_profiling_info<info::event_profiling::command_start>();
  cl_ulong time_end =
      e.get_profiling_info<info::event_profiling::command_end>();
  double elapsed = (time_end - time_start) / 1e6;
  std::cout << msg << k<<": "<< elapsed << " milliseconds\n";
}

// Forward declare the kernel names in the global scope. This FPGA best practice
// reduces compiler name mangling in the optimization reports.
template <int Replica> class Stencil;
//RunKernel(q, d_input, d_mask, d_output);
using FloatVector = std::vector<float,oneapi::tbb::cache_aligned_allocator<float>>; 

template <int Replica, int NumReplicas>
sycl::event RunKernel(queue& q, FloatVector& in, FloatVector& m,
               FloatVector& out){

    //[begin,end) traverses the updated rows in output 
    constexpr int begin = Replica * (kRows-2) / NumReplicas + 1;
    constexpr int end = (Replica+1) * (kRows-2) / NumReplicas + 1;
    // create the device buffers
    buffer b_input{in.begin()+(begin-1)*kCols, in.begin()+(end+1)*kCols, oneapi::tbb::cache_aligned_allocator<float>{}};
    buffer b_mask{m, oneapi::tbb::cache_aligned_allocator<float>{}};
    buffer b_output{out.begin()+begin*kCols, out.begin()+end*kCols, oneapi::tbb::cache_aligned_allocator<float>{}};
    b_output.set_final_data(out.begin()+begin*kCols);
    b_output.set_write_back();
    //printf("Replica: %d -> begin:%d; end:%d;   buffer_b:%d; buffer_e:%d\n", Replica,begin,end, (begin-1)*kCols, (end+1)*kCols);
    // submit the kernel
    auto e = q.submit([&](handler &h) {
      //Properties for HBM
      //ext::oneapi::accessor_property_list HBM0{ext::intel::buffer_location<Replica*2>};
      //ext::oneapi::accessor_property_list HBM0_noInit{no_init,ext::intel::buffer_location<Replica*2+1>};
      ext::oneapi::accessor_property_list HBM0{ext::intel::buffer_location<Replica>};
      ext::oneapi::accessor_property_list HBM0_noInit{no_init,ext::intel::buffer_location<Replica>};
      // Data accessors
      accessor input{b_input, h, /*read_only,*/ HBM0};
      accessor mask{b_mask, h, read_only};
      accessor output{b_output, h, write_only, HBM0_noInit};

      // Kernel executes with pipeline parallelism on the FPGA.
      // Use kernel_args_restrict to specify that input, mask, and output do not alias.
      h.single_task<Stencil<Replica>>([=]() [[intel::kernel_args_restrict]] {
          [[intel::fpga_register]]
          float local_mask[9];
          for(int i=0; i<9; i++) local_mask[i]=mask[i];

          // 3 shift registers for 3 rows
          [[intel::fpga_register]]
          fpga_tools::ShiftReg<float, 3> sr0;
          [[intel::fpga_register]]
          fpga_tools::ShiftReg<float, 3> sr1;
          [[intel::fpga_register]]
          fpga_tools::ShiftReg<float, 3> sr2;
          #pragma unroll
          for(int k=0; k<2; k++) {
            sr0.shiftSingleVal<1>(input[k]);
            sr1.shiftSingleVal<1>(input[k+kCols]);
            sr2.shiftSingleVal<1>(input[k+2*kCols]);
          }
          for(int i=kCols; i<(end-begin+1)*kCols; i++){
            int prow_base = i - kCols;
            int nrow_base = i + kCols;
            //Shift
            sr0.shiftSingleVal<1>(input[prow_base+2]);
            sr1.shiftSingleVal<1>(input[i+2]);
            sr2.shiftSingleVal<1>(input[nrow_base+2]);

            float tmp = local_mask[0] * sr0[0] + 
                        local_mask[1] * sr0[1] +
                        local_mask[2] * sr0[2] +
                        local_mask[3] * sr1[0] +
                        local_mask[4] * sr1[1] +
                        local_mask[5] * sr1[2] +
                        local_mask[6] * sr2[0] +
                        local_mask[7] * sr2[1] +
                        local_mask[8] * sr2[2] ;
            if(((i+1)&(kCols-1)) && ((i+2)&(kCols-1)) ){
              output[prow_base+1] = tmp;
            }
          }
      });
    });
    return e;
}

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
  FloatVector input(kArraySize); 
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

    // The definition of this function is in a different compilation unit,
    // so host and device code can be separately compiled.
    constexpr int NumRep=32;
    std::vector<sycl::event> events(NumRep);
    auto start = std::chrono::high_resolution_clock::now();
    fpga_tools::UnrolledLoop<NumRep>([&](auto k){
      events[k] = RunKernel<k,NumRep>(q, input, mask, output);});
    q.wait();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time FPGA: "<< std::chrono::duration<double,std::milli>(end - start).count() << " ms.\n";
    fpga_tools::UnrolledLoop<NumRep>([&](auto k){
      ReportTime("FPGA Stencil with HBM. Time IP ",k,events[k]);});

  } catch (exception const &e0) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e0.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e0.code().value() == CL_DEVICE_NOT_FOUND) {
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
