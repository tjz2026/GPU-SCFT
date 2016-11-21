/usr/opt/cuda/7.5/bin/nvcc  main.cpp init.cpp matrix.cpp field.cpp scft_driver.cpp init_cuda.cu cuda_scft.cu -o scft -lcufft -I /usr/opt/cuda/7.5/samples/common/inc/
