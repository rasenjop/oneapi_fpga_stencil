//==============================================================
// Vector Add is the equivalent of a Hello, World! sample for data parallel
// programs. Building and running the sample verifies that your development
// environment is setup correctly and demonstrates the use of the core features
// of DPC++. 
// Here we add to the original Vector Add support for HBM and kernel replication

// For comprehensive instructions regarding DPC++ Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <numeric>
#include <oneapi/tbb/cache_aligned_allocator.h>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

constexpr bool VERBOSE = false;
using namespace sycl;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

static void ReportTime(const std::string &msg, event e) {
  cl_ulong time_start =
      e.get_profiling_info<info::event_profiling::command_start>();
  cl_ulong time_end =
      e.get_profiling_info<info::event_profiling::command_end>();
  double elapsed = (time_end - time_start) / 1e6;
  std::cout << msg << elapsed << " milliseconds\n";
}

//************************************
// Vector add 
//************************************
// Vector type for this example.
using IntVector = std::vector<int,oneapi::tbb::cache_aligned_allocator<int>>; 

template <bool HBM_enabled, int Replica, int unroll_factor> class VAdd;

template<bool HBM_enabled, int Replica, int NumRep, int unroll_factor>
sycl::event VectorAdd(queue &q, const IntVector &a_vector, const IntVector &b_vector,
               IntVector &sum_parallel) {
  size_t num_items{a_vector.size()};

  int begin = Replica * num_items / NumRep;
  int end   = (Replica +1) * num_items / NumRep;
  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  buffer a_buf{a_vector.begin()+begin, a_vector.begin()+end, oneapi::tbb::cache_aligned_allocator<float>{}};
  if constexpr (VERBOSE){ 
    host_accessor temp_a{a_buf};
    for (int i=0; i<end-begin; i++) std::cout << temp_a[i] << ";  ";
    std::cout << "\n";
  } 
  buffer b_buf{b_vector.begin()+begin, b_vector.begin()+end, oneapi::tbb::cache_aligned_allocator<float>{}};
  if constexpr (VERBOSE){ 
    host_accessor temp_b{b_buf};
    for (int i=0; i<end-begin; i++) std::cout << temp_b[i] << ";  ";
    std::cout << "\n";
  }
  buffer sum_buf{sum_parallel.begin()+begin, sum_parallel.begin()+end, oneapi::tbb::cache_aligned_allocator<float>{}};
  sum_buf.set_final_data(sum_parallel.begin()+begin);
  sum_buf.set_write_back();
  if constexpr (VERBOSE){ 
    host_accessor temp_s{sum_buf};
    for (int i=0; i<end-begin; i++) std::cout << temp_s[i] << ";  ";
    std::cout << "\n";
  }
  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  auto e = q.submit([&](handler &h) {

    if constexpr (HBM_enabled){ 
      ext::oneapi::accessor_property_list PL0{ext::intel::buffer_location<Replica*3>};
      ext::oneapi::accessor_property_list PL1{ext::intel::buffer_location<Replica*3+1>};
      ext::oneapi::accessor_property_list PL2{no_init,ext::intel::buffer_location<Replica*3+2>};
      accessor a{a_buf, h, read_only, PL0};
      accessor b{b_buf, h, read_only, PL1};
      accessor sum{sum_buf, h, write_only, PL2};
      h.single_task<VAdd<HBM_enabled,Replica,unroll_factor>>([=]() [[intel::kernel_args_restrict]]{
        #pragma unroll unroll_factor
          for (size_t i = 0; i < end-begin; i++)
            sum[i] = a[i] + b[i]; 
      });
    }
    else{
      accessor a{a_buf, h, read_only};
      accessor b{b_buf, h, read_only};
      accessor sum{sum_buf, h, write_only, no_init};
      h.single_task<VAdd<HBM_enabled,Replica,unroll_factor>>([=]() [[intel::kernel_args_restrict]]{
        #pragma unroll unroll_factor
          for (size_t i = 0; i < end-begin; i++)
            sum[i] = a[i] + b[i]; 
      });
    }
  });
  if constexpr (VERBOSE){ 
    host_accessor temp_s{sum_buf};
    std::cout << "After computation \n";
    for (int i=0; i<end-begin; i++) std::cout << temp_s[i] << ";  ";
    std::cout << "\n";
  }
  return e;
}

//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char* argv[]) {
  // Change vector_size if it was passed as argument
  size_t vector_size = (argc > 1) ? std::stoi(argv[1]) : 1024;

  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  ext::intel::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  default_selector d_selector;
#endif

  // Create vector objects with "vector_size" to store the input and output data.
  IntVector a, b, sum_sequential, sum_parallel;
  a.resize(vector_size);
  b.resize(vector_size);
  sum_sequential.resize(vector_size);
  sum_parallel.resize(vector_size);

  // Initialize input vectors with values from 0 to vector_size - 1
  std::iota(a.begin(), a.end(), 0);
  std::iota(b.begin(), b.end(), 0);

  try {
    auto prop_list = property_list{property::queue::enable_profiling()};
    queue q(d_selector, exception_handler, prop_list);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Vector size: " << a.size() << "\n";

    // Vector addition in DPC++
    auto start = std::chrono::high_resolution_clock::now();
    auto e0 = VectorAdd<true,0,4,4>(q, a, b, sum_parallel);
    auto e1 = VectorAdd<true,1,4,4>(q, a, b, sum_parallel);
    auto e2 = VectorAdd<true,2,4,4>(q, a, b, sum_parallel);
    auto e3 = VectorAdd<true,3,4,4>(q, a, b, sum_parallel);
    q.wait();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time FPGA: "<< std::chrono::duration<double,std::milli>(end - start).count() << " ms.\n";

    ReportTime("FPGA VectorAdd on IP0 ",e0);
    ReportTime("FPGA VectorAdd on IP1 ",e1);
    ReportTime("FPGA VectorAdd on IP2 ",e2);
    ReportTime("FPGA VectorAdd on IP3 ",e3);


  } catch (exception const &e) {
      std::cout << "An exception is caught for vector add.\n";
      std::terminate();
  }

  // Compute the sum of two vectors in sequential for validation.
  for (size_t i = 0; i < sum_sequential.size(); i++)
    sum_sequential.at(i) = a.at(i) + b.at(i);

  // Verify that the two vectors are equal.  
  for (size_t i = 0; i < sum_sequential.size(); i++) {
    if (sum_parallel.at(i) != sum_sequential.at(i)) {
      std::cout << "Error at index i=" << i << " ; gold=" << sum_sequential.at(i) << "; out="<< sum_parallel.at(i) << '\n';
    }
  }

  int indices[]{0, 1, 2, (static_cast<int>(a.size()) - 1)};
  constexpr size_t indices_size = sizeof(indices) / sizeof(int);

  // Print out the result of vector add.
  for (int i = 0; i < indices_size; i++) {
    int j = indices[i];
    if (i == indices_size - 1) std::cout << "...\n";
    std::cout << "[" << j << "]: " << a[j] << " + " << b[j] << " = "
              << sum_parallel[j] << "\n";
  }

  a.clear();
  b.clear();
  sum_sequential.clear();
  sum_parallel.clear();
  return 0;
}
