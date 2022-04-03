// Copyright 2022 Novozhilov Alexander
#include <omp.h>
#include <vector>
#include <string>
#include <random>
#include <iostream>
#include "matrix_mult.h"

SparseMatrix::SparseMatrix(int _m, int _n) {
    std::vector<std::vector<std::complex<int>>> matrix = getRandomMatrix(_m, _n);
    m = matrix.size();
    n = matrix[0].size();
    std::complex<int> zero(0, 0);
    rowCounter.push_back(0);
    int counter = 0;
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[i].size(); j++) {
            if (matrix[i][j] != zero) {
                values.push_back(matrix[i][j]);
                columnIndexes.push_back(j);
                counter++;
            }
        }
        rowCounter.push_back(counter);
    }
}

SparseMatrix::SparseMatrix(std::vector<std::vector<std::complex<int>>> matrix) {
    m = matrix.size();
    n = matrix[0].size();
    std::complex<int> zero(0, 0);
    rowCounter.push_back(0);
    int counter = 0;
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[i].size(); j++) {
            if (matrix[i][j] != zero) {
                values.push_back(matrix[i][j]);
                columnIndexes.push_back(j);
                counter++;
            }
        }
        rowCounter.push_back(counter);
    }
}

SparseMatrix SparseMatrix::multiply_seq(const SparseMatrix& matrix) const {
    if (n != matrix.m) {
        throw std::invalid_argument("invalid matrix size");
    }
    std::complex<int> tmp;
    std::vector<std::vector<std::complex<int>>> resultMatrix = getEmptyMatrix(m, matrix.n);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < matrix.n; j++) {
            tmp = std::complex<int>();
            for (int k = 0; k < n; k++) {
                tmp += get(i, k) * matrix.get(k, j);
            }
            resultMatrix[i][j] = tmp;
        }
    }
    return SparseMatrix(resultMatrix);
}

int SparseMatrix::getM() const {
    return m;
}

int SparseMatrix::getN() const {
    return n;
}

std::complex<int> SparseMatrix::get(int row, int col) const {
    if (row > m || col > n || row < 0 || col < 0) {
        throw std::invalid_argument("index out of matrix range");
    }
    int currCol;
    for (size_t pos = rowCounter[row]; pos < rowCounter[row + 1]; pos++) {
        currCol = columnIndexes[pos];
        if (currCol == col) {
            return values[pos];
        }
    }
    return std::complex<int>();
}

void SparseMatrix::print() {
    printFullMatrix(getFullMatrix());
    std::cout << "values = ";
    for (size_t i = 0; i < values.size(); i++) {
        std::cout << values[i];
        std::cout << " ";
    }
    std::cout << std::endl;

    std::cout << "rowCounter = ";
    for (size_t i = 0; i < rowCounter.size(); i++) {
        std::cout << rowCounter[i];
        std::cout << " ";
    }
    std::cout << std::endl;

    std::cout << "columnIndexes = ";
    for (size_t i = 0; i < columnIndexes.size(); i++) {
        std::cout << columnIndexes[i];
        std::cout << " ";
    }
    std::cout << std::endl;
    std::cout << "____________________________" << std::endl;
}

std::vector<std::vector<std::complex<int>>> SparseMatrix::getRandomMatrix(int m, int n) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::vector<std::vector<std::complex<int>>> result;
    for (size_t i = 0; i < m; i++) {
        std::vector<std::complex<int>> vec;
        for (size_t j = 0; j < n; j++)
        {
            if (gen() % 100 <= 30) {
                std::complex<int> complexNumber(gen() % 10, (gen() % 10));
                vec.push_back(complexNumber);
            } else {
                std::complex<int> complexNumber(0, 0);
                vec.push_back(complexNumber);
            }
        }
        result.push_back(vec);
    }
    return result;
}

void SparseMatrix::printFullMatrix(std::vector<std::vector<std::complex<int>>> matrix) const {
    std::cout << "____________________________" << std::endl;
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[i].size(); j++) {
            std::cout << matrix[i][j]<< " ";
        }
        std::cout << std::endl;
    }
    std::cout << "____________________________" << std::endl;
}

std::vector<std::vector<std::complex<int>>> SparseMatrix::getFullMatrix() {
    std::vector<std::vector<std::complex<int>>> result;
    for (size_t i = 0; i < m; i++) {
        std::vector <std::complex<int>> tmp;
        for (size_t j = 0; j < n; j++) {
            tmp.push_back(get(i, j));
        }
        result.push_back(tmp);
    }
    return result;
}

std::vector<std::vector<std::complex<int>>> SparseMatrix::getEmptyMatrix(int m, int n) const {
    std::vector<std::vector<std::complex<int>>> matrix;
    for (size_t i = 0; i < m; i++) {
        matrix.push_back(std::vector<std::complex<int>>(n));
    }
    return matrix;
}
