// Copyright 2022 Kolesnikov Ilya
#include "../../../modules/task_3/kolesnikov_i_Cannon_dense_matrix/matrix.h"

void Matrix::generateMatrix(double num) {
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            matrix[i][j] = i*num;
        }
    }
}

std::vector< std::vector<double>> Matrix::shiftLeft(std::vector< std::vector<double>> *mat, size_t pos, size_t skew) {
    std::vector<double> tmp_matr(size);
    std::vector< std::vector<double>> vec = *mat;
    for (size_t i = 0; i < size; ++i) {
        tmp_matr[i] = vec[pos][(i + skew) % size];
    }
    for (size_t i = 0; i < size; ++i) {
        vec[pos][i] = tmp_matr[i];
    }
    return vec;
}

std::vector< std::vector<double>> Matrix::shiftUp(std::vector< std::vector<double>> *mat, size_t pos, size_t skew) {
    std::vector<double> tmp_matr(size);
    std::vector< std::vector<double>> vec = *mat;
    for (size_t i = 0; i < size; ++i) {
        tmp_matr[i] = vec[(i + skew) % size][pos];
    }
    for (size_t i = 0; i < size; ++i) {
        vec[i][pos] = tmp_matr[i];
    }
    return vec;
}

std::vector< std::vector<double>> Matrix::cannonAlgorithmSeq(Matrix matrix1, Matrix matrix2) {
    std::vector< std::vector<double>> res_matrix, matr1 = matrix1.matrix, matr2 = matrix2.matrix;
    for (size_t i = 0; i < size; ++i) {
        matr1 = shiftLeft(&matr1, i, i);
    }
    for (size_t j = 0; j < size; ++j) {
        matr2 = shiftUp(&matr2, j, j);
    }
    for (size_t i = 0; i < size; ++i) {
        std::vector<double> vec(size);
        res_matrix.push_back(vec);
        for (size_t j = 0; j < size; ++j) {
            res_matrix[i][j] = 0;
        }
    }
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            for (size_t k = 0; k < size; ++k) {
                res_matrix[j][k] += matr1[j][k] * matr2[j][k];
            }
        }
        for (size_t l = 0; l < size; ++l) {
            matr1 = shiftLeft(&matr1, l, 1);
        }
        for (size_t l = 0; l < size; ++l) {
            matr2 = shiftUp(&matr2, l, 1);
        }
    }
    return res_matrix;
}

std::vector< std::vector<double>> Matrix::cannonAlgorithmTBB(Matrix matrix1, Matrix matrix2, size_t thread_nums) {
    std::vector< std::vector<double>> res_matrix, matr1 = matrix1.matrix, matr2 = matrix2.matrix;

    for (size_t i = 0; i < size; ++i) {
        std::vector<double> vec(size);
        res_matrix.push_back(vec);
        for (size_t j = 0; j < size; ++j) {
            res_matrix[i][j] = 0;
        }
    }
    for (size_t i = 0; i < size; ++i) {
        matr1 = shiftLeft(&matr1, i, i);
    }
    for (size_t j = 0; j < size; ++j) {
        matr2 = shiftUp(&matr2, j, j);
    }
    for (size_t i = 0; i < size; ++i) {
        {
            tbb::parallel_for(tbb::blocked_range<int>(0, size), [&](const tbb::blocked_range<int>& range) {
                for (int j = range.begin(); j < range.end(); ++j) {
                    for (size_t k = 0; k < size; ++k) {
                        res_matrix[j][k] += matr1[j][k] * matr2[j][k];
                    }
                }
            });
        }
        for (size_t l = 0; l < size; ++l) {
            matr1 = shiftLeft(&matr1, l, 1);
        }
        for (size_t l = 0; l < size; ++l) {
            matr2 = shiftUp(&matr2, l, 1);
        }
    }
    return res_matrix;
}
