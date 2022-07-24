echo dpcpp cpu_kernel.cpp -c -o cpu_kernel.o -O3
dpcpp cpu_kernel.cpp -c -o cpu_kernel.o -O3
echo dpcpp -fintelfpga -DFPGA_EMULATOR -fsycl-link=image fpga_kernel.cpp -o fpga_kernel.a -ltbb -O3 -I ../common/
dpcpp -fintelfpga -DFPGA_EMULATOR -fsycl-link=image fpga_kernel.cpp -o fpga_kernel.a -ltbb -O3 -I ../common/
echo dpcpp -fintelfpga main.cpp cpu_kernel.o fpga_kernel.a -o main.fpga_emu  -ltbb 
dpcpp -fintelfpga main.cpp cpu_kernel.o fpga_kernel.a -o main.fpga_emu  -ltbb 