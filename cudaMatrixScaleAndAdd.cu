#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>

#define cudaCheckError() {                                                                       \
        cudaError_t e=cudaGetLastError();                                                        \
        if(e!=cudaSuccess) {                                                                     \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }

//Kernel
__global__ void scaleAndAdd(float *b, float *c, float *d, float a, int N) {
    
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i < N)
        d[i] = b[i] * a + c[i];
}

void initArray(float *arr, std::istream& inputStream , int n){
    for(int i = 0; i < n; i++){
        inputStream >> arr[i];
    }
}

int main(int argc, char* argv[]) {

    //check for number of arguments
    if(argc != 5){
        printf("Invalid number of arguments: %d", argc);
        return 0;
    }
    
    //parse arguments
    float a = atof(argv[1]);
    std::string bfile = argv[2];
    std::string cfile = argv[3];
    std::string dfile = argv[4];

    //open b and c stream
    std::ifstream bstream, cstream;
    bstream.open(bfile);
    cstream.open(cfile);

    //check if matrix addition is possible
    int brow, bcol, crow, ccol;
    bstream >> brow >> bcol;
    cstream >> crow >> ccol;
    if(brow != crow || bcol != ccol){
        printf("Invalid matrix size for addition: ");
        return 0;
    }

    
    int N = brow * bcol;
     
    

    int THREADS = 512;
    int BLOCKS = (N + THREADS - 1 ) / THREADS;

    //allocate memory.
    float *b, *c, *d;
    cudaMallocManaged(&b,N*sizeof(float));
    cudaMallocManaged(&c,N*sizeof(float));
    cudaMallocManaged(&d,N*sizeof(float));

    //initialize data
    initArray(b, bstream, N);
    initArray(c, cstream, N);
    bstream.close();
    cstream.close();

    //start Kernel
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now(); //time calculations start
    scaleAndAdd<<<BLOCKS,THREADS>>>(b, c, d, a, N);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now(); //time calculations end

    printf("Time taken: %lld\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()); //print time calculations

    //open d stream
    std::ofstream dstream;
    dstream.open(dfile);
    dstream << brow << " " << bcol << " ";
    for(int i=0;i<N;i++){
        dstream << d[i] << " ";
    }
    dstream << "\n";
    dstream.close();

    cudaFree(b);
    cudaFree(c);
    cudaFree(d);

    cudaCheckError();
    return 0;
}