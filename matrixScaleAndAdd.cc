#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include <chrono>

void printVector(std::vector<std::vector<float>> &inputVec){
    for(int i = 0; i < inputVec.size(); i++){
        std::vector<float> tvec = inputVec.at(i);
        for(int j = 0; j < tvec.size(); j++){
            printf("%.2f,", tvec.at(j));
        }
        printf("\n");
    }
}
int main(int argc, char *argv[]){
    if( argc == 5) {
        float a = atof(argv[1]);

        //create vector matrices
        std::vector<float> bVector;
        std::vector<float> cVector; 
        std::vector<float> dVector;

        //parse file b
        std::ifstream bFile;
        std::string bFileName = argv[2];
        bFile.open(bFileName);
        int brow, bcol;
        bFile >> brow >> bcol;
        //printf("row: %d|| col: %d\n", brow, bcol);
        float num;
        for(int i = 0; i < (brow * bcol); i++){

            bFile >> num;
            bVector.push_back(num);
        }
        //printVector(bVector);
        bFile.close();

        //parse file c
        std::ifstream cFile;
        std::string cFileName = argv[3];
        cFile.open(cFileName);
        int crow, ccol;
        cFile >> crow >> ccol;
        if(crow != brow && ccol != bcol) {
            printf("Invalid dimensions of matrices b: %d x %d c: %d x %d: ", brow, bcol, crow, ccol ); return 0;}
        //printf("row: %d|| col: %d\n", crow, ccol);
        for(int i = 0; i < (crow * ccol); i++){
            cFile >> num;
            cVector.push_back(num);
        }
        // printVector(cVector);
        cFile.close();

        //output calculations to d file
        std::string d = argv[4];
        std::ofstream dFile;
        dFile.open(d);
        dFile << brow << " " << bcol << " ";

        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now(); //time calculations start
        for(int i = 0; i < (brow * bcol); i++){
            float result = (a * bVector.at(i)) + cVector.at(i);
            dVector.push_back(result);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now(); //time calculations end
        
        for(int i = 0; i < dVector.size(); i++){
            dFile << dVector.at(i) << " ";
        }
        

        dFile << "\n";
        dFile.close();
        printf("Time taken: %lld\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }else{
        printf("Invalid Number of Arguments\n");
    }
    return 0;
}

