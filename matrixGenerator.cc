#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>

int main(int argc, char* argv[]){
    std::ofstream matrixb;
    std::ofstream matrixc;

    int m = atoi(argv[1]);
    matrixb.open("b");
    matrixc.open("c");
    matrixb << m << " " << m << " ";
    matrixc << m << " " << m << " ";
    for(long i = 0; i < m * m; i++){
        matrixb << rand() << " ";
        matrixc << rand() << " ";
    }
    matrixb.close();
    matrixc.close();
    return 0;

}