// Copyright 2022 Kolesnikov Ilya
#include "../../../modules/task_2/kolesnikov_i_Cannon_dense_matrix/matrix.h"

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

std::vector< std::vector<double>> Matrix::cannonAlgorithmOMP(Matrix matrix1, Matrix matrix2, size_t thread_nums) {
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
    size_t num_threads = thread_nums;
    size_t thread_num, block_size, j, k, start;
    for (size_t i = 0; i < size; ++i) {
#pragma omp parallel num_threads(num_threads) private(j, k, thread_num, start, block_size)
        {
            thread_num = omp_get_thread_num();
            block_size = size / num_threads;
            start = 0;
            if (thread_num != 0) {
                start = block_size * thread_num;
                if (num_threads < size) {
                    if (thread_num == num_threads - 1) {
                        block_size = size;
                    }
                    else {
                        block_size = start + block_size;
                    }
                }
                else if (num_threads > size) {
                    if (thread_num < size) {
                        start = thread_num;
                        block_size = thread_num + 1;
                    }
                }
                else {
                    block_size *= (thread_num + 1);
                }
            }
            if (num_threads > size && thread_num == 0) {
                block_size = 1;
            }
            for (j = start; j < block_size; ++j) {
                for (k = 0; k < size; ++k) {
                    res_matrix[j][k] += matr1[j][k] * matr2[j][k];
                }
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
