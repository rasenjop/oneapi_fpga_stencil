
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"
#include "unrolled_loop.hpp"
#include "shift_reg.hpp"
#include "constants.hpp"
using namespace sycl;

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

template <int Replica>
sycl::event RunKernel(queue& q, buffer<float>& b_input, buffer<float>& b_mask,
               buffer<float>& b_output, int begin, int end){



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

void run_fpga_kernel(FloatVector& in, FloatVector& m, FloatVector& out){
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

    constexpr int NumRep=32;
    std::vector<sycl::event> events(NumRep);
    //[begin,end) traverses the updated rows in output per kernel
    std::vector<int> begin(NumRep);
    std::vector<int> end(NumRep);
    for (int Replica=0; Replica<NumRep; Replica++){
      begin[Replica] = Replica * (kRows-2) / NumRep + 1;
      end[Replica] = (Replica+1) * (kRows-2) / NumRep + 1;
    }
    // create the device buffers
    //oneapi::tbb::cache_aligned_allocator<float> myAllocator{};
    buffer b_mask{m};
    std::vector<buffer<float>> b_input;
    std::vector<buffer<float>> b_output;
    auto start = std::chrono::high_resolution_clock::now();
    { //begin scope for buffer life-span
      fpga_tools::UnrolledLoop<NumRep>([&](auto k){
        b_input.push_back(buffer<float>{in.begin()+(begin[k]-1)*kCols, in.begin()+(end[k]+1)*kCols});
        b_output.push_back(buffer<float>{out.begin()+begin[k]*kCols, out.begin()+end[k]*kCols});
        //b_output.set_final_data(out.begin()+begin*kCols);
        //b_output.set_write_back();
      }); 
      
      //auto e0 = RunKernel<0>(q, b_input[0], b_mask, b_output[0], begin, end);
      fpga_tools::UnrolledLoop<NumRep>([&](auto k){
        events[k] = RunKernel<k>(q, b_input[k], b_mask, b_output[k], begin[k], end[k]);});
    } //Buffer destruction here
    q.wait();
    auto endt = std::chrono::high_resolution_clock::now();
    std::cout << "Time FPGA: "<< std::chrono::duration<double,std::milli>(endt - start).count() << " ms.\n";
    for (auto k=0; k<NumRep; k++){
      ReportTime("FPGA Stencil with HBM. Time IP ",k,events[k]);
    }

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
}