rm -rf stencil.fpga.prj
date
echo "Compile Host"
time dpcpp -fintelfpga -Xshardware -Xsboard=s10mx:p520_hpc_m210h_g3x16 host.cpp -o stencil.fpga -O3 -ltbb -I ../common/
date