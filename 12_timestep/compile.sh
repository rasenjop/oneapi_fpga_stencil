rm -rf fpga_kernel.a.prj fpga_kernel.a
date
echo "Compile CPU kernel"
echo dpcpp cpu_kernel.cpp -c -o cpu_kernel.o -O3
dpcpp cpu_kernel.cpp -c -o cpu_kernel.o -O3
echo "Compile FPGA kernel"
echo dpcpp -fintelfpga -fsycl-link=image -Xshardware -Xsboard=s10mx:p520_hpc_m210h_g3x16 fpga_kernel.cpp -o fpga_kernel.a -ltbb -O3 -I ../common/
time dpcpp -fintelfpga -fsycl-link=image -Xshardware -Xsboard=s10mx:p520_hpc_m210h_g3x16 fpga_kernel.cpp -o fpga_kernel.a -ltbb -O3 -I ../common/
echo "Link with main kernel"
echo dpcpp -fintelfpga main.cpp cpu_kernel.o fpga_kernel.a -o main.fpga -ltbb 
dpcpp -fintelfpga main.cpp cpu_kernel.o fpga_kernel.a -o main.fpga  -ltbb
date