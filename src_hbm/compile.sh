rm -rf dev_image.a dev_image.prj
echo "Compile Host"
dpcpp -fintelfpga -c host.cpp -o host.o -O3
echo "Compile Device"
time dpcpp -fintelfpga -Xshardware -Xsboard=s10mx:p520_hpc_m210h_g3x16 -fsycl-link=image kernel.cpp -o dev_image.a
echo "Link Host+Device"
dpcpp -fintelfpga host.o dev_image.a -o stencil.fpga -ltbb