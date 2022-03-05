all:
	nvcc cudaDeviceInfo.cu -O3 -o cudaDeviceInfo.out 
	nvcc matrixGenerator.cc -O3 -o matrixGenerator.out
	nvcc matrixScaleAndAdd.cc -O3 -o matrixScaleAndAdd.out  
	nvcc cudaMatrixScaleAndAdd.cu -O3 -o cudaMatrixScaleAndAdd.out
	nvcc greyscale.c -lSDL2 -lSDL2_image -o greyscale.out
	nvcc cudagreyscale.cu -lSDL2 -lSDL2_image -o cudagreyscale.out
	nvcc blurImage.c -lSDL2 -lSDL2_image -o blurImage.out
clean: 
	rm -f *.out
	rm -f *.lib
	rm -f *.exp
test:
	./cudaDeviceInfo.out
	./matrixGenerator.out 500
	./matrixScaleAndAdd.out 3.1 b c d
	./cudaMatrixScaleAndAdd.out 3.1 b c dm
	./greyscale.out testImage.png onethreadgrey.png
	./cudagreyscale.out testImage.png multithreadgrey.png
	./blurImage.out 10 testImage.png oneblurimage.png
