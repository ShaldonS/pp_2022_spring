// Copyright 2022 Kolesnikov Ilya
#include <gtest/gtest.h>
#include "./matrix.h"

TEST(CannonTest, test_1) {
    size_t size = 16;
    Matrix matrix1(size);
    Matrix matrix2(size);

    double num1 = 1.076, num2 = 2.067;
    matrix1.generateMatrix(num1);
    matrix2.generateMatrix(num2);

    double start, end;
    start = clock();
    Matrix matrix4(matrix1.cannonAlgorithmSeq(matrix1, matrix2), size);
    end = clock();
    std::cout.precision(3);
    double time_seq = (end - start) / CLOCKS_PER_SEC;
    std::cout << "\nSequence Time: " << time_seq << "\n";

    start = clock();
    Matrix matrix6(matrix1.cannonAlgorithmTBB(matrix1, matrix2, 2), size);
    end = clock();
    double time_tbb = (end - start) / CLOCKS_PER_SEC;
    std::cout << "\nParallel Time:" << time_tbb << "\n\n";

    std::vector<std::vector<double>> matr1 = matrix4.get_matrix(), matr2 = matrix6.get_matrix();

    ASSERT_DOUBLE_EQ(matr1[1][1], matr2[1][1]);
}


TEST(CannonTest, test_2) {
    size_t size = 4;
    Matrix matrix1(size);
    Matrix matrix2(size);

    double num1 = 1.076, num2 = 2.067;
    matrix1.generateMatrix(num1);
    matrix2.generateMatrix(num2);

    Matrix matrix4(matrix1.cannonAlgorithmSeq(matrix1, matrix2), size);

    Matrix matrix6(matrix1.cannonAlgorithmTBB(matrix1, matrix2, 2), size);

    std::vector<std::vector<double>> matr1 = matrix4.get_matrix(), matr2 = matrix6.get_matrix();
    ASSERT_DOUBLE_EQ(matr1[0][0], matr2[0][0]);
}

TEST(CannonTest, test_3) {
    size_t size = 16;
    Matrix matrix1(size);
    Matrix matrix2(size);

    double num1 = 1.076, num2 = 2.067;
    matrix1.generateMatrix(num1);
    matrix2.generateMatrix(num2);

    Matrix matrix4(matrix1.cannonAlgorithmSeq(matrix1, matrix2), size);

    Matrix matrix6(matrix1.cannonAlgorithmTBB(matrix1, matrix2, 2), size);

    std::vector<std::vector<double>> matr1 = matrix4.get_matrix(), matr2 = matrix6.get_matrix();
    ASSERT_DOUBLE_EQ(matr1[size - 1][size - 1], matr2[size - 1][size - 1]);
}

TEST(CannonTest, test_4) {
    size_t size = 9;
    Matrix matrix1(size);
    Matrix matrix2(size);

    double num1 = 1.076, num2 = 2.067;
    matrix1.generateMatrix(num1);
    matrix2.generateMatrix(num2);

    Matrix matrix4(matrix1.cannonAlgorithmSeq(matrix1, matrix2), size);

    Matrix matrix6(matrix1.cannonAlgorithmTBB(matrix1, matrix2, 2), size);

    std::vector<std::vector<double>> matr1 = matrix4.get_matrix(), matr2 = matrix6.get_matrix();

    for (size_t j(0); j < size; ++j) {
        ASSERT_DOUBLE_EQ(matr1[0][j], matr2[0][j]);
    }
}

TEST(CannonTest, test_5) {
    size_t size = 4;
    Matrix matrix1(size);
    Matrix matrix2(size);

    double num1 = 1.076, num2 = 2.067;
    matrix1.generateMatrix(num1);
    matrix2.generateMatrix(num2);

    Matrix matrix4(matrix1.cannonAlgorithmSeq(matrix1, matrix2), size);

    Matrix matrix6(matrix1.cannonAlgorithmTBB(matrix1, matrix2, 2), size);

    std::vector<std::vector<double>> matr1 = matrix4.get_matrix(), matr2 = matrix6.get_matrix();

    for (size_t i(0); i < size; ++i) {
        for (size_t j(0); j < size; ++j) {
            ASSERT_DOUBLE_EQ(matr1[i][j], matr2[i][j]);
        }
    }
}
