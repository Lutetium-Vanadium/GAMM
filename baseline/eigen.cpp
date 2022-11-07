#include <cstdint>
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
 
using Eigen::MatrixXf;

MatrixXf loadMatrix(char* path) {
    std::fstream f;
    f.open(path, std::ios_base::in | std::ios::binary);
    if (!f.is_open()) {
        std::cout << "Could not open " << path << std::endl;
        std::exit(1);
    }

    uint8_t buf[8];
    f.read((char*)&buf, 8);
    uint64_t n = *(uint64_t*)&buf;
    f.read((char*)&buf, 8);
    uint64_t m = *(uint64_t*)&buf;

    std::cout << n << ", " << m << std::endl;

    MatrixXf mat(n, m);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j ++) {
            uint8_t buf[4];
            f.read((char*)&buf, 4);
            mat(i, j) = *(float*)&buf;
        }
    }

    return mat;
}
 
int main() {
    MatrixXf x = loadMatrix("./baseline/x.dat");
    MatrixXf y = loadMatrix("./baseline/y.dat");
    auto t_start = std::chrono::high_resolution_clock::now();
    MatrixXf z = x*y.transpose();
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "x: " << x.block<2, 2>(0, 0) << std::endl;
    std::cout << "y: " << y.block<2, 2>(0, 0) << std::endl;
    std::cout << "z: " << z.block<2, 2>(0, 0) << std::endl;

    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    std::cout << elapsed_time_ms << " ms" << std::endl;
}
