#include "func.h"

#ifdef EIGEN
#include "Eigen/Dense"
#endif

#ifdef SGX
#include "../Enclave/Enclave.h"
#endif

#ifndef SGX
#include <cstdio>
#endif

void linear(float *input, float *weight, float *bias, float *output, int N,
            int indim, int outdim) {
#ifdef EIGEN
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        inputMatrix(input, N, indim);

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        weightMatrix(weight, outdim, indim);

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        biasMatrix(bias, 1, outdim);

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        outputMatrix(output, N, outdim);
    outputMatrix =
        inputMatrix * weightMatrix.transpose() + biasMatrix.replicate(N, 1);
#else
#ifdef DEBUG
    // printf("N: %d\n", N);
    // printf("indim: %d\n", indim);
    // printf("outdim: %d\n", outdim);
#endif
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < outdim; j++) {
            float sum = 0.0f;
            for (int k = 0; k < indim; k++) {
                sum += input[i * indim + k] * weight[j * indim + k];
#ifdef DEBUG
// printf("weight: %f\n", weight[j * indim + k]);
// printf("input: %f\n", input[i * indim + k]);
// printf("sum: %f\n", sum);
#endif
            }
            output[i * outdim + j] = sum + bias[j];
        }
    }
#endif
}