// Copyright 2022 Kolesnikov Ilya
#ifndef MODULES_TASK_3_KOLESNIKOV_I_CANNON_DENSE_MATRIX_MATRIX_H_
#define MODULES_TASK_3_KOLESNIKOV_I_CANNON_DENSE_MATRIX_MATRIX_H_
#include <vector>
#include <iostream>
#include "tbb/tbb.h"

class Matrix {
 public:
    explicit Matrix(size_t size) :size(size) {
        matrix.reserve(size);
        for (size_t i(0); i < size; ++i) {
            std::vector<double> vec(size);
            matrix.push_back(vec);
        }
    }
    Matrix(std::vector<std::vector<double>> matrix, size_t size) :size(size) {
        this->matrix.reserve(size);
        for (size_t i(0); i < size; ++i) {
            std::vector<double> vec;
            this->matrix.push_back(vec);
        }
        for (size_t i(0); i < size; ++i) {
            this->matrix[i].assign(matrix[i].begin(), matrix[i].end());
        }
    }
    ~Matrix() {}
    std::vector<std::vector<double>> get_matrix() {
        return matrix;
    }
    void generateMatrix(double num);
    std::vector<std::vector<double>> multiplyByMatrix(Matrix matrix);
    std::vector<std::vector<double>> cannonAlgorithmSeq(Matrix matrix1, Matrix matrix2);
    std::vector< std::vector<double>> shiftLeft(std::vector<std::vector<double>> *mat, size_t pos, size_t skew);
    std::vector< std::vector<double>> shiftUp(std::vector<std::vector<double>> *mat, size_t pos, size_t skew);
    std::vector< std::vector<double>> cannonAlgorithmTBB(Matrix matrix1, Matrix matrix2, size_t thread_nums);

 private:
    size_t size;
    std::vector<std::vector<double>> matrix;
};
#endif  // MODULES_TASK_3_KOLESNIKOV_I_CANNON_DENSE_MATRIX_MATRIX_H_