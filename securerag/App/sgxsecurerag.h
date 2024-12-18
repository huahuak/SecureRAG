#ifndef SGXSECURERAG_H
#define SGXSECURERAG_H

void initSGX();

void sgxSecureLinear(float *input, float *weight, float *bias, float *output,
                     int N, int indim, int outdim);

#endif