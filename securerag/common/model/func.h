#ifndef FUNC_H
#define FUNC_H

void embedding();

void linear(float *input, float *weight, float *bias, float *output, int N,
            int indim, int outdim);

#endif