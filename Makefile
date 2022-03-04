all: cudaDeviceInfo.cu 
	nvcc cudaDeviceInfo.cu -O3 -o cudaDeviceInfo.out 
	nvcc matrixGenerator.cc -O3 -o matrixGenerator.out
	nvcc matrixScaleAndAdd.cc -O3 -o matrixScaleAndAdd.out  
	nvcc cudaMatrixScaleAndAdd.cu -O3 -o cudaMatrixScaleAndAdd.out
clean: 
	rm -f *.out
	rm -f *.lib
	rm -f *.exp
test:
	./cudaDeviceInfo.out
	./matrixGenerator.out 1000
	./matrixScaleAndAdd.out 3.1 b c d
	./cudaMatrixScaleAndAdd.out 3.1 b c dm
