dpcpp -fintelfpga -DFPGA_EMULATOR host.cpp kernel.cpp -o stencil.fpga_emu -O3 -ltbb