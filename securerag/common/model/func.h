#ifndef FUNC_H
#define FUNC_H

void embedding();

void linear(float *input, float *weight, float *bias, float *output, int N,
            int indim, int outdim);

void attention(float *q, float *k, float *out, float *qw, float *qb, float *kw,
               float *kb, float *vw, float *vb, float *fw, float *fb, int bsz,
               int tgtlen, int srclen, int embeddim, int nh);

#endif