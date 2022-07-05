rm -rf dev_image.a dev_image.prj
echo "Compile Host"
time dpcpp -fintelfpga host.cpp -O3 -Xshardware -Xsboard=s10mx:p520_hpc_m210h_g3x16 -fsycl-link=image -o stencil_fpga -fbracket-depth=3000 -ltbb
