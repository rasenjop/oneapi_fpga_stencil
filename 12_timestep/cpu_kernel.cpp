
#include <CL/sycl.hpp>
#include "dpc_common.hpp"
#include "constants.hpp"
using namespace sycl;

static void ReportTimeCPU(const std::string &msg, event e) {
  cl_ulong time_start =
      e.get_profiling_info<info::event_profiling::command_start>();
  cl_ulong time_end =
      e.get_profiling_info<info::event_profiling::command_end>();
  double elapsed = (time_end - time_start) / 1e6;
  std::cout << msg << ": "<< elapsed << " milliseconds\n";
}

void run_cpu_kernel(FloatVector& in, FloatVector& m, FloatVector& out){
      // Select either the FPGA emulator or FPGA device

  try {
    cpu_selector device_selector;
    auto prop_list = property_list{property::queue::enable_profiling()};
    queue q(device_selector, dpc_common::exception_handler, prop_list);

    range<2> alloc_range(kRows, kCols);
    range<2> stencil_range(kRows-2, kCols-2);
    auto start = std::chrono::high_resolution_clock::now();
    buffer<float,2> b_input{in.data(),alloc_range};
    buffer b_mask{m};
    buffer<float,2> b_output{out.data(),alloc_range};
    // submit the kernel
    auto e = q.submit([&](handler &h) {
      // Data accessors
      accessor input{b_input, h, read_only};
      accessor mask{b_mask, h, read_only};
      accessor output{b_output, h, write_only, no_init};

      // Kernel executes with pipeline parallelism on the FPGA.
      // Use kernel_args_restrict to specify that input, mask, and output do not alias.
      h.parallel_for(stencil_range, [=](id<2> idx){
        int i=idx[0]+1;
        int j=idx[1]+1;
        float tmp = mask[0] * input[i-1][j-1] + 
                    mask[1] * input[i-1][ j ] +
                    mask[2] * input[i-1][j+1] +
                    mask[3] * input[ i ][j-1] +
                    mask[4] * input[ i ][ j ] +
                    mask[5] * input[ i ][j+1] +
                    mask[6] * input[i+1][j-1] +
                    mask[7] * input[i+1][ j ] +
                    mask[8] * input[i+1][j+1] ;
        output[i][j] = tmp;
      });
    });
    q.wait();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Total time CPU sycl: "<< std::chrono::duration<double,std::milli>(end - start).count() << " ms.\n";
    ReportTimeCPU("CPU Stencil in SYCL. Kernel Time",e);
  } catch (exception const &ex) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << ex.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (ex.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "Please ensure that your CPU device is properly configured.\n";
    }
    std::terminate();
  }
}