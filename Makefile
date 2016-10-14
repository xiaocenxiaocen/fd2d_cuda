# Location of the CUDA Toolkit
#CUDA_PATH = $(MJONIR)/opt/cuda
CUDA_PATH = /home/zx/Software/cuda-7.5
CC = icc -O3 -g -Wall -std=c99
CXX = icpc -O3 -g -Wall -std=c++0x -Wno-deprecated

NVCC = nvcc -ccbin gcc -Xcompiler -fopenmp

#NVCC = nvcc -ccbin icc -Xcompiler -openmp

CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_COMMON_INCLUDE = $(CUDA_PATH)/samples/common/inc
INCLUDES = -I$(CUDA_COMMON_INCLUDE) -I$(CUDA_INCLUDE) 

GENCODE_FLAGS = -m64 -gencode arch=compute_20,code=sm_20
CUDA_FLAGS = --ptxas-options=-v
CFLAGS = $(CUDA_FLAGS)

LIBRARIES = -L.

LDFLAGS = -lm -lpthread

all: target

target: get_gpu_info \
fd2d_gpu_test

get_gpu_info.o: get_gpu_info.cpp
	$(NVCC) $(INCLUDES) $(CFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

get_gpu_info: get_gpu_info.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

fd2d_gpu_kernel.o: fd2d_gpu_kernel.cu
	$(NVCC) $(INCLUDES) $(CFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

fd2d_gpu_test: fd2d_gpu_kernel.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)


.PHONY: clean
clean:
	-rm *.o
	-rm get_gpu_info
	-rm fd2d_gpu_test
