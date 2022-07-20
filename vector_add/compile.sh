rm -rf vector-add-repl.fpga.prj vector-add-repl.fpga
date
echo "Compile Host"
time dpcpp -fintelfpga -Xshardware -Xsboard=s10mx:p520_hpc_m210h_g3x16 -DFPGA=1 vector-add-repl.cpp -o vadd.fpga -O3 -ltbb
date