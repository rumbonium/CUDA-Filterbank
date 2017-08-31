# CUDA-Filterbank
Using CUDA and a Filterbank method to filter all 100 FM stations simultaneously

Filterbank program:
  Included libraries (the libraries that need to be installed to run this program):
  <fcntl.h>
  <sys/stat.h>
  <sys/types.h>
  <unistd.h>
  <stdio.h>
  <cuda_runtime.h>
  <stdlib.h>
  <math.h>
  <string.h>
  <cufft.h>
  <sys/ipc.h>
  <sys/shm.h>
  <sys/sem.h>
  <complex>
  Also install Sox

  Included files (keep these files in the same directory as filtbank3.cu):
  "filtb.h"
  "filtd.h"
  "filt7.h"
  
  Compile Instructions:
  Compile command: "nvcc filtbank3.cu -o filtbank3 -lcufft"
  
  Execute Instructions: Run this program first. The 'stdout' of this program is piped to sox for audio.
  Execute command: "./filtbank3 <radio station (i.e. 92.9)> | play --rate 40k -b 32 -c 1 -e float -t raw -"
  

Usrp stream program:
  Included libraries (the libraries that need to be installed to run this program):
  <uhd/types/tune_request.hpp>
  <uhd/utils/thread_priority.hpp>
  <uhd/utils/safe_main.hpp>
  <uhd/usrp/multi_usrp.hpp>
  <uhd/exception.hpp>
  <boost/program_options.hpp>
  <boost/format.hpp>
  <boost/thread.hpp>
  <iostream>
  <fstream>
  <csignal>
  <complex>
  <fcntl.h>
  <sys/stat.h>
  <sys/types.h>
  <unistd.h>
  <algorithm>
  <sys/types.h>
  <sys/ipc.h>
  <sys/shm.h>
  <sys/sem.h>
  <stdio.h>
  <stdlib.h>
  <unistd.h>
  <string.h>
  <stdlib.h>

  Compile Instructions:
  Compile command: "usrp_stream4.cpp -o usrp_stream4 -luhd -lboost_system -lboost_thread -lboost_program_options"
  
  Execute Instructions: Run this program second
  Execute command: "./usrp_stream4.cpp --rate 20000000 --freq 98000000 --spb 16384 --type float"
