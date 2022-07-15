
#include <CL/sycl.hpp>
#include "constants.hpp"
using namespace sycl;

void run_cpu_kernel(FloatVector& in, FloatVector& m, FloatVector& out){
      // Select either the FPGA emulator or FPGA device

  ext::intel::cpu_selector device_selector;
  try {
    auto prop_list = property_list{property::queue::enable_profiling()};
    queue q(device_selector, dpc_common::exception_handler, prop_list);

    auto start = std::chrono::high_resolution_clock::now();
    buffer b_input{in};
    buffer b_mask{m};
    buffer b_output{out};
    // submit the kernel
    auto e = q.submit([&](handler &h) {
      // Data accessors
      accessor input{b_input, h, read_only};
      accessor mask{b_mask, h, read_only};
      accessor output{b_output, h, write_only, noInit};

      // Kernel executes with pipeline parallelism on the FPGA.
      // Use kernel_args_restrict to specify that input, mask, and output do not alias.
      h.parallel_for(1UL, static_cast<unsigned long>(kRows-1), [&](int i){
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
    }).wait();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time CPU: "<< std::chrono::duration<double,std::milli>(end - start).count() << " ms.\n";

  } catch (exception const &e0) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e0.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "Please ensure that your CPU device is properly configured.\n";
    }
    std::terminate();
  }
}