#include "func.h"
#ifdef SGX
#include "../Enclave/Enclave.h"
#endif
#ifndef SGX
#include <cstdio>
#endif

void linear(float *input, float *weight, float *bias, float *output, int N,
            int indim, int outdim) {
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
}