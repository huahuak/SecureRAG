#ifndef SGXSECURERAG_H
#define SGXSECURERAG_H

#include "core.h"

void initSGX();

TensorRef sgxCopyTensorToSGX(Tensor &tensor);

Tensor sgxCopyTensorFromSGX(TensorRef ref);

void sgxSecureLinear(float *input, float *weight, float *bias, float *output,
                     int N, int indim, int outdim);

void sgxSecureAttention(float *q, float *k, float *out, float *qw, float *qb,
                        float *kw, float *kb, float *vw, float *vb, float *fw,
                        float *fb, int bsz, int tgtlen, int srclen,
                        int embeddim, int nh);

void sgxSecureBMM(float *m1, float *m2, int B, int len, int dim);

#endif