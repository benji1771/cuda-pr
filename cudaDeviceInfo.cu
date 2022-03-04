#include <cstdio>
#include <cstdlib>


int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("Device Count: %d \n", deviceCount);
  for(int i = 0; i < deviceCount; i++){
      cudaSetDevice(i);


      cudaDeviceProp devProp;
      cudaGetDeviceProperties(&devProp, i);
      //Get Version
      int version;
      cudaDriverGetVersion(&version);
      printf("Device Version: %d \n", version);
      //global memory
      printf("Global Memory: %zu \n", devProp.totalGlobalMem);
      //constant mem size
      printf("Constant Memory: %zu \n", devProp.totalConstMem);
      //shared mem per block
      printf("Shared Memory Per Block: %zu \n", devProp.sharedMemPerBlock);
      //max block dimensions
      //printf("Max Block Per MultiProcessor: %d \n", devProp.maxBlocksPerMultiProcessor);
      //number of multiprocessors
      printf("Number of Multiprocessors: %d \n", devProp.multiProcessorCount);
      //max grid dimensions
      printf("Max Grid Size: %d \n", devProp.maxGridSize[3]);
      //warp size
      printf("Warp Size: %d \n", devProp.warpSize);
      
  }
  return 0;
}
