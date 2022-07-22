
# Modern C++17 features to optimize FPGA SYCL applications

We propose a simple running example (based on a stencil computation) that is iteratively improved using FPGA specific optimizations, such us loop unrolling, kernel replication, High Bandwidth Memory (HBM) modules exploitation and the Shift Register Pattern (SRP). To ease the implementation of these optimizations we leverage C++17 language features as metaprogramming and templates. We follow an incremental strategy to attack the learning curve and an informative approach with plenty of examples. We conduct the experiments on a Stratix 10 MX with 32 HBM modules (from BittWare) that we have configured recently so that it can be targeted by oneAPI. 



***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* RHEL*/CentOS* 8
| Hardware                          | Intel® Stratix 10 MX with 32 HBM modules <br> Intel® FPGA 3rd party / custom platforms with oneAPI support 
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | C++17 idioms and optimizations for FPGA using the stencil computation as a running example
| Time to complete                  | 1 hour


## Purpose

To get a FPGA optimized implementation the code should be as simple as possible, it should maximize the amount of information that is known at compile-time and it should use some idioms that the compiler can effectively transform into "HW" (the bitstream), like the Shift Register Pattern (SRP). This translates into cluttered source codes with repeated statements (redundancies) and error prone constructions that become a challenge in terms of maintainability. Counterintuitively, using **high-level** C++17 features (like template metaprogramming, extensive use of "constexpr" and "if constexpr", variadic templates, lambdas, non-type template parameters, etc.) can result in very efficient **low-level** FPGA implementations that at the same time reduce the programming effort and increase code maintainability.

Tanner Young-Schultz recently delivered an inspiring [webinar](https://www.intel.com/content/www/us/en/developer/videos/adaptive-noise-reduction-design-using-oneapi.html) targeting an image filtering app (ANR, Adaptive Noise Reduction) for FPGAs in which he leverages modern C++17 programming in order to ease some FPGA optimizations. The code is available on [GitHub](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2BFPGA/ReferenceDesigns/anr) (as part of the oneAPI examples). However, the ANR application is more complicated than necessary in order to illustrate the C++17 idioms and optimizations. Here we follows Tanner's approach and use some of his libraries (loop unrolling, shift register, etc), but using a simpler example and adding some additional optimizations like kernel replication for data-level parallelism exploitation.

This tutorial explains the evolution of a canonic stencil algorithm and the required transformations needed to be efficiently executed on an FPGA. To follow this evolution, please have a look a the following directory structure:

### Optimizing the `stencil` code for FPGA devices

0. Naive CPU-like implementation (**00_naive**)



1. Baseline FPGA implementation (**01_baseline**)

2. Plus High Bandwidth Memory enabled (**02_hbm**)


3. Plus Shift Register Pattern optimization (**03_srp**)


4. Plus Loop Unrolling (pragma) (**04_srp_unroll**)


5. Plus Loop Unrolling (template) (**05_srp_template**)


6. Plus Data-level kernel replication (**06_kreplic**)


7. Plus Alternative Shift Register (**07_newsrp_kreplic**)


8. Plus Loop Flattening (**08_1loop**)

9. Compare with Parallel CPU implementation in TBB (**09_parallel_for**)


10. Compare with Parallel CPU impl. in SYCL (**10_sycl_cpu**)

      SYCL Implementation of a similar stencil code [here](https://github.com/Apress/data-parallel-CPP/blob/main/samples/Ch14_common_parallel_patterns/fig_14_14_stencil.cpp)

11. Right version of 08_1loop that avoids kernel serialization (**11_buffer**)
   See [this forum post](https://community.intel.com/t5/Application-Acceleration-With/Data-level-parallelism-on-FPGA-with-kernel-replication-using/td-p/1397792) for a detailed explanation.



### **Performance on FPGA Stratix 10 MX:**

| Version                     | Throughput (M elements/sec)
|---                          |---
| 00_naive                    | 0.892  
| 01_baseline                 | 47  
| 02_hbm                      | 174  
| 03_srp                      | 0.07
| 04_srp_unroll               | 132  
| 05_srp_template             | 132  
| 06_kreplic   (2 kernels)    | 166 
| 07_newsrp_kreplic (16 ker, 2HBM/ker) | 4,142  
| 08_1loop (32 ker, 1HBM/ker) | 12,427
| 11_buffer (32 ker, 1HBM/ker) | 6,100


### **Performance on CPU Core i7-7820X (8 cores - 2th/core):**

| Version                     | Throughput (M elements/sec)
|---                          |---
| 09_parallel_for (TBB) | 5,162 
| 10_sycl_cpu (SYCL)   | 3,947 

### **Emulation and Compilation instructions:** (work in progress)

For emulation (FPGA emulation on CPU) use

```
# Emulation
./emu.sh
./stencil.fpga_emu
```

For the real execution on the FPGA use
```
# FPGA Compilation
./compile.sh
./stencil.fpga
```

With more detail, the compilation is a 3-step process:

1. Compile the device code:

   ```
   dpcpp -fintelfpga -fsycl-link=image kernel.cpp -o dev_image.a -Xshardware -Xsboard=s10mx:p520_hpc_m210h_g3x16
   ```
   Input files should include all source files that contain device code. This step may take several hours. The `-Xsboard` flag states the FPGA board we are using (Stratix 10MX)


2. Compile the host code:

   ```
   dpcpp -fintelfpga host.cpp -c -o host.o
   ```
   Input files should include all source files that only contain host code. This takes seconds.


3. Create the device link:

   ```
   dpcpp -fintelfpga host.o dev_image.a -o stencil.fpga
   ```
   The input should have N (N >= 0) host object files *(.o)* and one device image file *(.a)*. This takes seconds.

**NOTE:** You only need to perform steps 2 and 3 when modifying host-only files.


## Key Concepts
* FPGA kernel designs require specific optimizations tailores to the FPGA architecture
* C++17 features ease the development of such optimizations
* One of the key takeaways of this tutorial is that leveraging the main C++ design principles (like "design for for easy extension" --aka Open-Closed principle--, "separation of concerns", --aka Single-responsibility principle--, and "simplify change principle", aka --Don't repeat yourself principle--) are actually valid and more than advisable for FPGA programming.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)


## Building the `stencil` Tutorial

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud (not tested yet)
If running a sample in the Intel DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.



### Work in progress: On a Linux* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:
    ```
    cmake ..
   ```
   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake .. -DFPGA_BOARD=<board-support-package>:<board-variant>
   ```

     **NOTE:** For the FPGA emulator target and the FPGA target, the device link method is used.
2. Compile the design through the generated `Makefile`. The following build targets are provided:

   * Compile for emulation (fast compile time, targets emulated FPGA device):
      ```
      make fpga_emu
      ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     make fpga
     ```
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/fast_recompile.fpga.tar.gz" download>here</a>.



### Example of Output (10_sycl_cpu)
```
FPGA Stencil with HBM. Time IP 0: 2.73709 milliseconds                         
FPGA Stencil with HBM. Time IP 1: 2.59427 milliseconds                         
FPGA Stencil with HBM. Time IP 2: 2.57754 milliseconds                         
FPGA Stencil with HBM. Time IP 3: 2.60263 milliseconds                         
FPGA Stencil with HBM. Time IP 4: 2.58751 milliseconds                         
FPGA Stencil with HBM. Time IP 5: 2.60341 milliseconds                         
FPGA Stencil with HBM. Time IP 6: 2.59619 milliseconds                         
FPGA Stencil with HBM. Time IP 7: 2.59615 milliseconds                         
FPGA Stencil with HBM. Time IP 8: 2.60307 milliseconds                         
FPGA Stencil with HBM. Time IP 9: 2.59579 milliseconds                         
FPGA Stencil with HBM. Time IP 10: 2.59859 milliseconds                        
FPGA Stencil with HBM. Time IP 11: 2.5954 milliseconds                         
FPGA Stencil with HBM. Time IP 12: 2.6003 milliseconds                         
FPGA Stencil with HBM. Time IP 13: 2.59353 milliseconds                        
FPGA Stencil with HBM. Time IP 14: 2.60412 milliseconds                        
FPGA Stencil with HBM. Time IP 15: 2.59862 milliseconds                        
FPGA Stencil with HBM. Time IP 16: 2.5882 milliseconds                         
FPGA Stencil with HBM. Time IP 17: 2.55117 milliseconds                        
FPGA Stencil with HBM. Time IP 18: 2.59665 milliseconds                        
FPGA Stencil with HBM. Time IP 19: 2.5979 milliseconds                         
FPGA Stencil with HBM. Time IP 20: 2.60135 milliseconds                        
FPGA Stencil with HBM. Time IP 21: 2.59885 milliseconds                        
FPGA Stencil with HBM. Time IP 22: 2.59378 milliseconds                        
FPGA Stencil with HBM. Time IP 23: 2.59281 milliseconds                        
FPGA Stencil with HBM. Time IP 24: 2.59014 milliseconds                        
FPGA Stencil with HBM. Time IP 25: 2.60636 milliseconds                        
FPGA Stencil with HBM. Time IP 26: 2.59406 milliseconds                        
FPGA Stencil with HBM. Time IP 27: 2.60779 milliseconds                        
FPGA Stencil with HBM. Time IP 28: 2.59006 milliseconds                        
FPGA Stencil with HBM. Time IP 29: 2.58648 milliseconds                        
FPGA Stencil with HBM. Time IP 30: 2.59117 milliseconds                        
FPGA Stencil with HBM. Time IP 31: 2.58266 milliseconds                        
Total time CPU sycl: 108.431 ms.
CPU Stencil in SYCL. Kernel Time: 8.5971 milliseconds                          
Time CPU seq: 27.5616 ms.
Time CPU tbb::parallel_for: 6.56578 ms.
PASSED: results are correct

```

### Discussion of Results
Try modifying `host.cpp` to produce a different output message. Then, perform a host-only recompile via the device link method to see how quickly the design is recompiled.
